"""用 TQC + HER 训练 Aloha Pick & Place。

在项目根目录运行。向量环境的创建和 ``learn`` 都在 ``main`` 里，
``main`` 只由 ``if __name__ == "__main__"`` 调用，满足 Windows 下 SubprocVecEnv 的要求。

正式训练的默认总步数是 1_000_000。冒烟测试请显式传入更小的 ``--total-timesteps``。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import aloha_envs  # noqa: F401  注册 AlohaPickAndPlaceSparse-v0
import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.her import HerReplayBuffer

from scripts.callbacks import (
    CheckpointLatestCallback,
    SuccessEvalCallback,
    ThroughputCallback,
    VideoEvalCallback,
    buffer_transition_count,
)

TB_LOG_NAME = "TQC"
EVAL_SEED_OFFSET = 10_000
VIDEO_SEED_OFFSET = 20_000
VERSION_PACKAGES = (
    "torch",
    "stable-baselines3",
    "sb3-contrib",
    "mujoco",
    "gymnasium",
    "gymnasium-robotics",
)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def make_pap_env(
    env_id: str,
    seed: int,
    p_air: float,
    time_feature: bool,
    render_mode: str | None,
):
    """gym.make（含 TimeLimit）→ 可选 TimeFeatureWrapper → Monitor。"""

    def _init():
        import aloha_envs  # noqa: F401

        kwargs: dict = {"p_air": float(p_air)}
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        env = gym.make(env_id, **kwargs)
        if time_feature:
            env = TimeFeatureWrapper(env)
        env = Monitor(env)
        env.reset(seed=int(seed))
        return env

    return _init


def _package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in VERSION_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_commit() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return f"unavailable: {exc}"
    return output.strip()


def _wrapper_order(time_feature: bool) -> list[str]:
    order = ["gym.make(TimeLimit)"]
    if time_feature:
        order.append("TimeFeatureWrapper")
    order.append("Monitor")
    return order


def _close_quietly(env) -> None:
    """关闭环境。子进程已经退出时不再在管道上收发，避免卡住。"""
    if env is None:
        return
    processes = list(getattr(env, "processes", None) or [])
    child_dead = False
    for process in processes:
        try:
            child_dead = child_dead or not process.is_alive()
        except Exception:
            child_dead = True
    if child_dead:
        try:
            env.waiting = False
        except Exception:
            pass
        for process in processes:
            try:
                if process.is_alive():
                    process.kill()
            except Exception:
                pass
        for process in processes:
            try:
                process.join(timeout=2)
            except Exception:
                pass
        try:
            env.closed = True
        except Exception:
            pass
        return
    try:
        env.close()
    except Exception:
        pass
    for process in processes:
        try:
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        except Exception:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TQC + HER 训练 Aloha Pick & Place",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-id", default="AlohaPickAndPlaceSparse-v0")
    parser.add_argument("--p-air", type=float, default=0.5)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=("subproc", "dummy"), default="subproc")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--n-critics", type=int, default=2)
    parser.add_argument("--n-sampled-goal", type=int, default=4)
    parser.add_argument("--goal-selection-strategy", default="future")
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=None,
        help="省略时等于 n_envs，使每个环境步对应一次梯度更新。",
    )
    parser.add_argument("--time-feature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-freq-total", type=int, default=20_000)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--save-freq-total", type=int, default=100_000)
    parser.add_argument("--video-every-n-evals", type=int, default=1)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume-model", default="")
    parser.add_argument("--resume-buffer", default="")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args(argv)


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_name:
        return ROOT / "runs" / args.run_name
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return ROOT / "runs" / f"tqc_{args.env_id}_{stamp}"


def _build_config(args: argparse.Namespace, train_env, eval_freq: int, save_freq: int) -> dict:
    max_episode_steps = int(train_env.get_attr("_max_episode_steps")[0])
    distance_threshold = float(train_env.get_attr("distance_threshold")[0])
    p_air = float(train_env.get_attr("p_air")[0])
    time_feature_steps = None
    if args.time_feature:
        time_feature_steps = int(train_env.get_attr("_max_steps")[0])
    train_seeds = list(train_env._seeds)
    gradient_steps = args.n_envs if args.gradient_steps is None else args.gradient_steps
    return {
        "args": vars(args),
        "resolved": {
            "gradient_steps": gradient_steps,
            "eval_freq": eval_freq,
            "save_freq": save_freq,
            "eval_freq_total": args.eval_freq_total,
            "save_freq_total": args.save_freq_total,
            "policy": "MultiInputPolicy",
            "replay_buffer_class": "HerReplayBuffer",
            "handle_timeout_termination": True,
            "tb_log_name": TB_LOG_NAME,
        },
        "env": {
            "env_id": args.env_id,
            "p_air": p_air,
            "distance_threshold": distance_threshold,
            "max_episode_steps": max_episode_steps,
            "time_feature": bool(args.time_feature),
            "time_feature_max_steps": time_feature_steps,
            "wrapper_order": _wrapper_order(bool(args.time_feature)),
            "observation_space": str(train_env.observation_space),
            "action_space": str(train_env.action_space),
            "video": {
                "render_mode": "rgb_array",
                "width": 320,
                "height": 240,
                "camera": "eval_cam",
                "fps": 25,
            },
        },
        "seeds": {
            "seed": args.seed,
            "train": train_seeds,
            "eval_base": args.seed + EVAL_SEED_OFFSET,
            "eval_episode_rule": "eval_base + episode_index，每次评估重复同一组",
            "video": args.seed + VIDEO_SEED_OFFSET,
        },
        "versions": _package_versions(),
        "git_commit": _git_commit(),
        "resumed_from": {
            "model": args.resume_model or None,
            "buffer": args.resume_buffer or None,
        },
    }


def _write_config(run_dir: Path, config: dict, resume: bool) -> Path:
    if resume and (run_dir / "config.json").exists():
        path = run_dir / "config_resume.json"
    else:
        path = run_dir / "config.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _timeout_fraction(model) -> float | None:
    buffer = getattr(model, "replay_buffer", None)
    if buffer is None or not hasattr(buffer, "timeouts"):
        return None
    used = int(buffer.size())
    if used <= 0:
        return 0.0
    return float(np.mean(buffer.timeouts[:used]))


def _save_final(model, run_dir: Path) -> dict[str, float]:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_dir / "final_model.zip"
    step_path = checkpoint_dir / f"{int(model.num_timesteps)}_steps.zip"
    buffer_path = checkpoint_dir / "replay_buffer_latest.pkl"
    started = time.perf_counter()
    model.save(str(final_path))
    model.save(str(step_path))
    model_seconds = time.perf_counter() - started
    started = time.perf_counter()
    model.save_replay_buffer(str(buffer_path))
    buffer_seconds = time.perf_counter() - started
    print(
        f"FINAL step={int(model.num_timesteps)} final_model={final_path.name} "
        f"checkpoint={step_path.name} buffer_transitions={buffer_transition_count(model.replay_buffer)} "
        f"timeout_fraction={_timeout_fraction(model)}",
        flush=True,
    )
    return {"final_model_seconds": model_seconds, "final_buffer_seconds": buffer_seconds}


def main(argv: list[str] | None = None) -> None:
    _configure_stdio()
    args = parse_args(argv)
    if args.n_envs < 1:
        raise SystemExit("--n-envs 必须 >= 1")
    if args.total_timesteps < 1:
        raise SystemExit("--total-timesteps 必须 >= 1")
    eval_freq = args.eval_freq_total // args.n_envs
    save_freq = args.save_freq_total // args.n_envs
    if eval_freq < 1 or save_freq < 1:
        raise SystemExit("eval_freq_total 和 save_freq_total 都必须 >= n_envs")
    if args.eval_freq_total % args.n_envs != 0 or args.save_freq_total % args.n_envs != 0:
        print(
            "警告：频率不能被 n_envs 整除，已按整数除法截断。"
            f" eval_freq={eval_freq} save_freq={save_freq}",
            flush=True,
        )
    gradient_steps = args.n_envs if args.gradient_steps is None else args.gradient_steps
    run_dir = _resolve_run_dir(args)
    if run_dir.exists() and not args.resume_model:
        raise SystemExit(f"运行目录已存在：{run_dir}。请更换 --run-name，或用 --resume-model 续训。")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)
    print(f"RUN_DIR={run_dir}", flush=True)

    vec_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv
    train_env = vec_cls(
        [
            make_pap_env(
                args.env_id,
                args.seed + rank,
                args.p_air,
                bool(args.time_feature),
                None,
            )
            for rank in range(args.n_envs)
        ]
    )
    eval_env = make_pap_env(
        args.env_id,
        args.seed + EVAL_SEED_OFFSET,
        args.p_air,
        bool(args.time_feature),
        None,
    )()
    video_env = make_pap_env(
        args.env_id,
        args.seed + VIDEO_SEED_OFFSET,
        args.p_air,
        bool(args.time_feature),
        "rgb_array",
    )()
    model = None
    throughput = ThroughputCallback(verbose=0)
    interrupted = False
    save_error: BaseException | None = None
    try:
        replay_buffer_kwargs = {
            "n_sampled_goal": args.n_sampled_goal,
            "goal_selection_strategy": args.goal_selection_strategy,
            "handle_timeout_termination": True,
        }
        policy_kwargs = {"net_arch": list(args.net_arch), "n_critics": args.n_critics}
        if args.resume_model:
            model = TQC.load(
                args.resume_model,
                env=train_env,
                device=args.device,
                tensorboard_log=str(run_dir / "tb"),
            )
            if args.resume_buffer:
                model.load_replay_buffer(args.resume_buffer)
            if model.replay_buffer is not None and int(model.replay_buffer.n_envs) != args.n_envs:
                raise SystemExit(
                    "续训的 replay buffer n_envs="
                    f"{model.replay_buffer.n_envs}，当前 --n-envs={args.n_envs}。"
                )
            reset_num_timesteps = False
            print(
                f"RESUME num_timesteps={int(model.num_timesteps)} "
                f"buffer_transitions={buffer_transition_count(model.replay_buffer)} "
                f"handle_timeout_termination={model.replay_buffer.handle_timeout_termination}",
                flush=True,
            )
        else:
            model = TQC(
                policy="MultiInputPolicy",
                env=train_env,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                tau=args.tau,
                gamma=args.gamma,
                train_freq=args.train_freq,
                gradient_steps=gradient_steps,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=replay_buffer_kwargs,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(run_dir / "tb"),
                verbose=args.verbose,
                seed=args.seed,
                device=args.device,
            )
            reset_num_timesteps = True
        if not isinstance(model.replay_buffer, HerReplayBuffer):
            raise RuntimeError(f"replay buffer 不是 HerReplayBuffer：{type(model.replay_buffer)}")
        if not model.replay_buffer.handle_timeout_termination:
            raise RuntimeError("HerReplayBuffer.handle_timeout_termination 必须为 True")

        # set_random_seed 在构造 / load 时用的是原始种子。续训改成
        # seed + 已完成步数，避免重放第一次训练开头的物块和目标。
        if args.resume_model:
            train_env.seed(args.seed + int(model.num_timesteps))
        else:
            train_env.seed(args.seed)
        config = _build_config(args, train_env, eval_freq, save_freq)
        config_path = _write_config(run_dir, config, resume=bool(args.resume_model))
        print(f"CONFIG={config_path}", flush=True)
        print(
            f"FREQ eval_freq={eval_freq} save_freq={save_freq} gradient_steps={model.gradient_steps} "
            f"wrappers={config['env']['wrapper_order']} train_seeds={config['seeds']['train']}",
            flush=True,
        )

        callbacks = [
            SuccessEvalCallback(
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                eval_seed=args.seed + EVAL_SEED_OFFSET,
                run_dir=run_dir,
                verbose=args.verbose,
            ),
            VideoEvalCallback(
                video_env=video_env,
                eval_freq=eval_freq,
                video_every_n_evals=args.video_every_n_evals,
                video_seed=args.seed + VIDEO_SEED_OFFSET,
                run_dir=run_dir,
                verbose=args.verbose,
            ),
            CheckpointLatestCallback(save_freq=save_freq, run_dir=run_dir, verbose=args.verbose),
            throughput,
        ]
        try:
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=callbacks,
                log_interval=args.log_interval,
                tb_log_name=TB_LOG_NAME,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=False,
            )
        except (KeyboardInterrupt, EOFError, BrokenPipeError, ConnectionResetError) as exc:
            interrupted = True
            print(
                f"捕获到 {type(exc).__name__}，正在保存 final_model 和最新的 replay buffer。",
                flush=True,
            )
        try:
            final_timing = _save_final(model, run_dir)
        except (Exception, KeyboardInterrupt) as exc:
            save_error = exc
            print(f"保存失败：{type(exc).__name__}: {exc}", flush=True)
        else:
            perf = throughput.finish()
            perf.update(final_timing)
            perf["timeout_fraction"] = _timeout_fraction(model)
            perf["buffer_transitions"] = buffer_transition_count(model.replay_buffer)
            perf["eval_seconds"] = list(callbacks[0].eval_seconds)
            perf["video_seconds"] = list(callbacks[1].video_seconds)
            perf["video_bytes"] = list(callbacks[1].video_bytes)
            perf["checkpoint_model_seconds"] = list(callbacks[2].model_save_seconds)
            perf["checkpoint_buffer_seconds"] = list(callbacks[2].buffer_save_seconds)
            buffer_file = run_dir / "checkpoints" / "replay_buffer_latest.pkl"
            perf["buffer_file_bytes"] = buffer_file.stat().st_size if buffer_file.exists() else None
            zips = sorted((run_dir / "checkpoints").glob("*_steps.zip"), key=lambda path: path.stat().st_mtime)
            perf["checkpoint_file_bytes"] = zips[-1].stat().st_size if zips else None
            perf["final_model_bytes"] = (run_dir / "final_model.zip").stat().st_size
            perf_name = "perf_resume.json" if args.resume_model else "perf.json"
            perf_path = run_dir / perf_name
            perf_path.write_text(json.dumps(perf, indent=2), encoding="utf-8")
            print(f"PERF_JSON={perf_path}", flush=True)
            gpu_reserved = perf["gpu_peak_reserved_bytes"]
            gpu_text = "none" if gpu_reserved is None else f"{gpu_reserved / (1024 ** 2):.1f}"
            pre_sps = perf["pre_sps"]
            post_sps = perf["post_sps_wall"]
            pre_text = "none" if pre_sps is None else f"{pre_sps:.1f}"
            post_text = "none" if post_sps is None else f"{post_sps:.1f}"
            print(
                "PERF "
                f"pre_sps={pre_text} post_sps_wall={post_text} "
                f"steps={perf['num_timesteps_start']}->{perf['num_timesteps_end']} "
                f"ram_peak_mb={perf['ram_peak_bytes'] / (1024 ** 2):.1f} "
                f"gpu_reserved_mb={gpu_text}",
                flush=True,
            )
    finally:
        _close_quietly(train_env)
        _close_quietly(eval_env)
        _close_quietly(video_env)
    if save_error is not None:
        raise SystemExit(1)
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
