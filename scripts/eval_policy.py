"""独立评估已训练的 Aloha Pick & Place 策略。

在项目根目录运行。环境用 ``scripts/train_tqc_pap.py`` 的 ``make_pap_env`` 构造，
回合循环用 ``scripts/callbacks.py`` 的 ``evaluate_episodes``。不启动训练，
不把环境传给 ``TQC.load``，也不加载 replay buffer。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sb3_contrib import TQC

from scripts.callbacks import _goal_is_air, evaluate_episodes, record_episode_video
from scripts.train_tqc_pap import _configure_stdio, _git_commit, _package_versions, make_pap_env

SUCCESS_THRESHOLD = 0.8
VIDEO_MAX_BYTES = 500 * 1024


class _ReplayBufferNotLoaded:
    """占位对象，不是磁盘上的 replay buffer。

    SB3 2.9 的 ``TQC.load`` 不传 env 时，仍会按保存的 ``HerReplayBuffer``
    新建一块空 buffer，而这个类的构造要求 env 不是 None。传入非 None 占位后，
    ``_setup_model`` 会跳过 buffer 分配。评估只调用 ``predict``，不读取该对象，
    也不调用 ``load_replay_buffer``。
    """
_WRAPPER_LABELS = {
    "TimeLimit": "gym.make(TimeLimit)",
    "TimeFeatureWrapper": "TimeFeatureWrapper",
    "Monitor": "Monitor",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="独立评估 TQC Pick & Place 策略",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", nargs="+", required=True, help="模型 zip 路径，可以写多个")
    parser.add_argument(
        "--config",
        default=None,
        help="config.json 路径。省略时取模型所在运行目录下的 config.json",
    )
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=30000, help="回合 i 使用 seed_base + i")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-videos", type=int, default=0, help="录制前 N 个回合的视频")
    parser.add_argument(
        "--render-human",
        action="store_true",
        help="评估环境使用 human 渲染。只实现该模式，不要在无人值守时打开窗口",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="输出目录。多个模型时，每个模型写到该目录下的 <模型名>_seed<seed_base>/",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="输出目录已存在且非空时仍然写入",
    )
    return parser.parse_args(argv)


def _run_dir_for_model(model_path: Path) -> Path:
    parent = model_path.parent
    if parent.name in {"best_model", "checkpoints"}:
        return parent.parent
    return parent


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"无法解析 JSON：{path}：{exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"JSON 顶层必须是对象：{path}")
    return data


def _blocked_seeds(config: dict[str, Any]) -> dict[str, list[int]]:
    """训练、训练中评估、视频用过的种子。公式见 M5 任务说明 3.1 第 3 条。"""
    try:
        seed = int(config["seeds"]["seed"])
        n_envs = int(config["args"]["n_envs"])
        n_eval = int(config["args"]["n_eval_episodes"])
        eval_base = int(config["seeds"]["eval_base"])
        video = int(config["seeds"]["video"])
        listed = [int(item) for item in config["seeds"]["train"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"config.json 的种子字段无法读取：{exc}") from exc
    if n_envs < 1 or n_eval < 1:
        raise SystemExit("config.json 里的 n_envs 和 n_eval_episodes 都必须 >= 1")
    train = [seed + index for index in range(n_envs)]
    if listed != train:
        raise SystemExit(
            "config.json 的 seeds.train 与 seed + 0..n_envs-1 不一致："
            f" listed={listed} formula={train}"
        )
    return {
        "train": train,
        "eval": [eval_base + index for index in range(n_eval)],
        "video": [video],
    }


def _check_seed_overlap(blocked: dict[str, list[int]], seed_base: int, n_episodes: int) -> list[int]:
    used = {seed for group in blocked.values() for seed in group}
    requested = set(range(seed_base, seed_base + n_episodes))
    overlap = sorted(used & requested)
    if overlap:
        raise SystemExit(
            "评估种子与训练已经用过的种子重叠。"
            f" 请求 [{seed_base}, {seed_base + n_episodes})，重叠 {overlap}。"
            f" 已占用 train={blocked['train']} eval={blocked['eval']} video={blocked['video']}。"
        )
    return sorted(requested)


def _default_out_dir(run_dir: Path, model_path: Path, seed_base: int) -> Path:
    return run_dir / "eval_policy" / f"{model_path.stem}_seed{seed_base}"


def _resolve_out_dir(args: argparse.Namespace, run_dir: Path, model_path: Path, n_models: int) -> Path:
    automatic = _default_out_dir(run_dir, model_path, args.seed_base)
    if args.out_dir is None:
        return automatic
    base = Path(args.out_dir).expanduser()
    if n_models == 1:
        return base
    return base / automatic.name


def _ensure_out_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(f"输出目录已存在且非空：{path}。若要覆盖请传 --overwrite。")
    path.mkdir(parents=True, exist_ok=True)


def _observed_wrapper_order(env) -> list[str]:
    labels_outside_in: list[str] = []
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        label = _WRAPPER_LABELS.get(type(current).__name__)
        if label is not None:
            labels_outside_in.append(label)
        current = getattr(current, "env", None)
    labels_outside_in.reverse()
    return labels_outside_in


def _env_params(config: dict[str, Any]) -> dict[str, Any]:
    env_cfg = config.get("env")
    if not isinstance(env_cfg, dict):
        raise SystemExit("config.json 缺少 env 对象")
    try:
        params = {
            "env_id": str(env_cfg["env_id"]),
            "p_air": float(env_cfg["p_air"]),
            "time_feature": bool(env_cfg["time_feature"]),
            "wrapper_order": [str(item) for item in env_cfg["wrapper_order"]],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"config.json 的环境字段无法读取：{exc}") from exc
    return params


def _build_env(params: dict[str, Any], seed: int, render_mode: str | None):
    env = make_pap_env(
        params["env_id"],
        seed,
        params["p_air"],
        params["time_feature"],
        render_mode,
    )()
    observed = _observed_wrapper_order(env)
    if observed != params["wrapper_order"]:
        env.close()
        raise SystemExit(
            "环境 wrapper 顺序与 config.json 不一致。"
            f" config={params['wrapper_order']} observed={observed}"
        )
    return env, observed


def _observation_dim(space) -> int:
    spaces = getattr(space, "spaces", None)
    if not isinstance(spaces, dict) or "observation" not in spaces:
        raise SystemExit(f"观测空间不是含 observation 键的 Dict：{space}")
    shape = spaces["observation"].shape
    if len(shape) != 1:
        raise SystemExit(f"observation 不是一维：{space}")
    return int(shape[0])


def _check_observation_space(model, env, time_feature: bool) -> int:
    expected = 26 if time_feature else 25
    env_dim = _observation_dim(env.observation_space)
    model_dim = _observation_dim(model.observation_space)
    if env.observation_space != model.observation_space or env_dim != expected or model_dim != expected:
        raise SystemExit(
            "模型观测空间与环境不一致，或 observation 维度不是"
            f" {expected}。\n"
            f"env ({env_dim}): {env.observation_space}\n"
            f"model ({model_dim}): {model.observation_space}"
        )
    return env_dim


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate(count: int, total: int) -> float:
    if total == 0:
        return float("nan")
    return float(count) / float(total)


def _fmt_count(count: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{count}/{total} ({count / total:.3f})"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _jsonable(float(value))
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


def _episode_table(result: dict[str, Any]) -> dict[str, Any]:
    successes = np.asarray(result["successes"])
    holds = np.asarray(result["holds"])
    distances = np.asarray(result["final_distances"], dtype=np.float64)
    is_air = np.asarray(result["is_air"], dtype=bool)
    n_episodes = int(successes.shape[0])
    success_count = int(np.sum(successes >= 0.5))
    hold_count = int(np.sum(holds >= 0.5))
    ground = ~is_air
    air = is_air
    n_ground = int(np.sum(ground))
    n_air = int(np.sum(air))
    n_ground_success = int(np.sum(successes[ground] >= 0.5)) if n_ground else 0
    n_air_success = int(np.sum(successes[air] >= 0.5)) if n_air else 0
    return {
        "n_episodes": n_episodes,
        "success_count": success_count,
        "success_rate": _rate(success_count, n_episodes),
        "hold_count": hold_count,
        "hold_rate": _rate(hold_count, n_episodes),
        "n_ground": n_ground,
        "n_ground_success": n_ground_success,
        "success_rate_ground": _rate(n_ground_success, n_ground),
        "n_air": n_air,
        "n_air_success": n_air_success,
        "success_rate_air": _rate(n_air_success, n_air),
        "final_distance_mean": float(result["final_distance"]),
        "final_distance_median": float(np.median(distances)) if n_episodes else float("nan"),
        "final_distance_max": float(np.max(distances)) if n_episodes else float("nan"),
        "mean_reward": float(result["mean_reward"]),
        "seconds": float(result["seconds"]),
        "meets_threshold": success_count * 100 >= n_episodes * int(SUCCESS_THRESHOLD * 100),
        "evaluate_episodes_success_rate": float(result["success_rate"]),
        "evaluate_episodes_hold_rate": float(result["hold_rate"]),
        "evaluate_episodes_success_rate_ground": float(result["success_rate_ground"]),
        "evaluate_episodes_success_rate_air": float(result["success_rate_air"]),
    }


def _record_videos(model, result: dict[str, Any], seed_base: int, n_videos: int, out_dir: Path, params: dict[str, Any]) -> list[dict[str, Any]]:
    if n_videos <= 0:
        return []
    video_env, _order = _build_env(params, seed_base, "rgb_array")
    records: list[dict[str, Any]] = []
    try:
        video_dir = out_dir / "videos"
        for index in range(n_videos):
            seed = seed_base + index
            air = bool(result["is_air"][index])
            kind = "air" if air else "ground"
            target = video_dir / f"seed{seed}_{kind}.mp4"
            probe, _info = video_env.reset(seed=int(seed))
            probe_air = bool(_goal_is_air(video_env, probe["desired_goal"]))
            info = record_episode_video(model, video_env, seed, target)
            csv_success = bool(float(result["successes"][index]) >= 0.5)
            video_success = bool(info["success"])
            record = {
                "seed": seed,
                "is_air": air,
                "video_is_air": probe_air,
                "air_match": air == probe_air,
                "path": str(info["path"]),
                "file_name": Path(info["path"]).name,
                "success": video_success,
                "csv_success": csv_success,
                "success_match": video_success == csv_success,
                "bytes": int(info["bytes"]),
                "under_500kb": int(info["bytes"]) < VIDEO_MAX_BYTES,
                "quality": int(info["quality"]),
                "n_frames": int(info["n_frames"]),
                "seconds": float(info["seconds"]),
            }
            records.append(record)
            match_text = "match" if record["success_match"] and record["air_match"] else "MISMATCH"
            print(
                "VIDEO "
                f"seed={seed} file={record['file_name']} bytes={record['bytes']} "
                f"video_success={int(video_success)} csv_success={int(csv_success)} "
                f"air={int(air)} video_air={int(probe_air)} {match_text}",
                flush=True,
            )
    finally:
        video_env.close()
    return records


def _write_episodes(path: Path, result: dict[str, Any], seed_base: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["seed", "is_air", "success", "hold", "final_distance", "reward"])
        n_episodes = int(len(result["successes"]))
        for index in range(n_episodes):
            writer.writerow(
                [
                    seed_base + index,
                    int(bool(result["is_air"][index])),
                    int(float(result["successes"][index]) >= 0.5),
                    int(float(result["holds"][index]) >= 0.5),
                    float(result["final_distances"][index]),
                    float(result["rewards"][index]),
                ]
            )


def _print_failures(result: dict[str, Any], seed_base: int) -> None:
    failed = 0
    for index, flag in enumerate(np.asarray(result["successes"])):
        if float(flag) >= 0.5:
            continue
        failed += 1
        air = int(bool(result["is_air"][index]))
        distance = float(result["final_distances"][index])
        print(
            f"FAIL seed={seed_base + index} is_air={air} final_distance={distance:.6f}",
            flush=True,
        )
    if failed == 0:
        print("FAIL none", flush=True)


def _print_result(model_path: Path, stats: dict[str, Any]) -> None:
    print(
        "RESULT "
        f"model={model_path.name} "
        f"success={_fmt_count(stats['success_count'], stats['n_episodes'])} "
        f"hold={_fmt_count(stats['hold_count'], stats['n_episodes'])} "
        f"ground={_fmt_count(stats['n_ground_success'], stats['n_ground'])} "
        f"air={_fmt_count(stats['n_air_success'], stats['n_air'])} "
        f"dist_mean={stats['final_distance_mean']:.6f} "
        f"dist_median={stats['final_distance_median']:.6f} "
        f"dist_max={stats['final_distance_max']:.6f} "
        f"reward={stats['mean_reward']:.4f} "
        f"seconds={stats['seconds']:.2f}",
        flush=True,
    )


def _rank_key(stats: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(stats["success_rate"]),
        float(stats["hold_rate"]),
        -float(stats["final_distance_mean"]),
    )


def _print_summary_table(rows: list[dict[str, Any]]) -> None:
    best_key = max(_rank_key(row) for row in rows)
    best_names = [row["model_name"] for row in rows if _rank_key(row) == best_key]
    print("SUMMARY", flush=True)
    header = (
        f"{'model':<24} {'success':<16} {'hold':<16} {'ground':<16} "
        f"{'air':<16} {'dist_mean':>10} {'dist_median':>12} {'dist_max':>10} {'reward':>8} {'seconds':>8}"
    )
    print(header, flush=True)
    for row in rows:
        marker = "*" if row["model_name"] in best_names else " "
        print(
            f"{row['model_name']:<24} "
            f"{_fmt_count(row['success_count'], row['n_episodes']):<16} "
            f"{_fmt_count(row['hold_count'], row['n_episodes']):<16} "
            f"{_fmt_count(row['n_ground_success'], row['n_ground']):<16} "
            f"{_fmt_count(row['n_air_success'], row['n_air']):<16} "
            f"{row['final_distance_mean']:>10.6f} "
            f"{row['final_distance_median']:>12.6f} "
            f"{row['final_distance_max']:>10.6f} "
            f"{row['mean_reward']:>8.3f} "
            f"{row['seconds']:>8.2f} {marker}",
            flush=True,
        )
    print(
        "BEST "
        + ", ".join(best_names)
        + " （成功率最高；相同则比较保持率；再相同则平均最终距离更小）",
        flush=True,
    )


def _prepare_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.n_episodes < 1:
        raise SystemExit("--n-episodes 必须 >= 1")
    if args.n_videos < 0:
        raise SystemExit("--n-videos 不能为负")
    if args.n_videos > args.n_episodes:
        raise SystemExit("--n-videos 不能大于 --n-episodes")
    jobs: list[dict[str, Any]] = []
    for model_arg in args.model:
        model_path = Path(model_arg).expanduser()
        if not model_path.is_file():
            raise SystemExit(f"找不到模型文件：{model_path}")
        model_path = model_path.resolve()
        run_dir = _run_dir_for_model(model_path)
        if args.config:
            config_path = Path(args.config).expanduser()
        else:
            config_path = run_dir / "config.json"
        if not config_path.is_file():
            raise SystemExit(f"找不到 config.json：{config_path}")
        config_path = config_path.resolve()
        config = _load_json(config_path)
        params = _env_params(config)
        blocked = _blocked_seeds(config)
        _check_seed_overlap(blocked, args.seed_base, args.n_episodes)
        out_dir = _resolve_out_dir(args, run_dir, model_path, len(args.model))
        jobs.append(
            {
                "model_path": model_path,
                "run_dir": run_dir,
                "config_path": config_path,
                "config": config,
                "params": params,
                "blocked": blocked,
                "out_dir": out_dir,
            }
        )
    out_dirs = [job["out_dir"] for job in jobs]
    if len(set(out_dirs)) != len(out_dirs):
        raise SystemExit(f"多个模型解析到了同一个输出目录：{out_dirs}")
    for job in jobs:
        out_dir = job["out_dir"]
        if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
            raise SystemExit(f"输出目录已存在且非空：{out_dir}。若要覆盖请传 --overwrite。")
    return jobs


def _evaluate_job(
    job: dict[str, Any],
    args: argparse.Namespace,
    versions: dict[str, str],
    git_commit: str,
) -> dict[str, Any]:
    params = job["params"]
    model_path: Path = job["model_path"]
    out_dir: Path = job["out_dir"]
    render_mode = "human" if args.render_human else None
    print(
        "ENV "
        f"model={model_path} config={job['config_path']} "
        f"env_id={params['env_id']} p_air={params['p_air']} "
        f"time_feature={params['time_feature']} wrapper_order={params['wrapper_order']} "
        f"render_mode={render_mode} device={args.device} "
        f"seed_base={args.seed_base} n_episodes={args.n_episodes} out_dir={out_dir}",
        flush=True,
    )
    print(
        "SEEDS "
        f"range=[{args.seed_base}, {args.seed_base + args.n_episodes}) "
        f"train={job['blocked']['train']} eval={job['blocked']['eval']} "
        f"video={job['blocked']['video']} overlap=0",
        flush=True,
    )
    env, observed = _build_env(params, args.seed_base, render_mode)
    model = None
    try:
        # 不传 env，也不调用 load_replay_buffer。占位见 _ReplayBufferNotLoaded。
        model = TQC.load(
            str(model_path),
            device=args.device,
            replay_buffer=_ReplayBufferNotLoaded(),
        )
        obs_dim = _check_observation_space(model, env, params["time_feature"])
        print(f"OBS_DIM {obs_dim} space_match=1", flush=True)
        result = evaluate_episodes(model, env, args.n_episodes, args.seed_base)
        stats = _episode_table(result)
        _print_result(model_path, stats)
        _print_failures(result, args.seed_base)
        _ensure_out_dir(out_dir, args.overwrite)
        video_started = time.perf_counter()
        videos = _record_videos(model, result, args.seed_base, args.n_videos, out_dir, params)
        video_seconds = time.perf_counter() - video_started if args.n_videos else 0.0
        payload = {
            "model_path": str(model_path),
            "model_name": model_path.name,
            "model_sha256": _sha256(model_path),
            "config_path": str(job["config_path"]),
            "env_params": params,
            "wrapper_order_observed": observed,
            "distance_threshold": job["config"]["env"].get("distance_threshold"),
            "max_episode_steps": job["config"]["env"].get("max_episode_steps"),
            "training_git_commit": job["config"].get("git_commit"),
            "seed_base": args.seed_base,
            "n_episodes": args.n_episodes,
            "seed_check": job["blocked"],
            "device": args.device,
            "loaded_device": str(getattr(model, "device", args.device)),
            "observation_dim": obs_dim,
            "versions": versions,
            "git_commit": git_commit,
            "videos": videos,
            "video_seconds": video_seconds,
            "success_threshold": SUCCESS_THRESHOLD,
        }
        payload.update(stats)
        summary_path = out_dir / "summary.json"
        episodes_path = out_dir / "episodes.csv"
        _write_episodes(episodes_path, result, args.seed_base)
        summary_path.write_text(
            json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"WROTE summary={summary_path} episodes={episodes_path}", flush=True)
        return payload
    finally:
        env.close()
        if model is not None:
            del model


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = parse_args(argv)
    versions = _package_versions()
    git_commit = _git_commit()
    print(f"VERSIONS {json.dumps(versions, ensure_ascii=False)} git={git_commit}", flush=True)
    jobs = _prepare_jobs(args)
    rows: list[dict[str, Any]] = []
    for job in jobs:
        rows.append(_evaluate_job(job, args, versions, git_commit))
    if len(rows) > 1:
        _print_summary_table(rows)
    below = [row["model_name"] for row in rows if not row["meets_threshold"]]
    if below:
        print(f"EXIT 1 success_below_80={below}", flush=True)
        return 1
    print("EXIT 0 每个模型的成功率都 >= 80%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
