"""TQC 训练用的评估、视频、checkpoint 和吞吐量回调。

评估自己跑回合，不调用 ``evaluate_policy``，这样每一步的 ``is_success`` 都能留下。
视频在评估之后录，checkpoint 的 replay buffer 只覆盖写最新的一份。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import psutil
import torch
from stable_baselines3.common.callbacks import BaseCallback

# 贴地目标的 z 等于物块静止高度；空中目标在此之上。
# 环境里空中高度是 ground + U(0, target_height_max)，等于 ground 的概率可忽略。
AIR_GOAL_MARGIN_M = 1e-6
HOLD_STEPS = 10
VIDEO_MAX_BYTES = 500 * 1024
VIDEO_FPS = 25


def _as_float(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _is_success_flag(value: Any) -> bool:
    return _as_float(value) > 0.5


def _goal_is_air(env, desired_goal: np.ndarray) -> bool:
    ground_z = float(env.unwrapped.initial_object_height)
    goal_z = float(np.asarray(desired_goal, dtype=np.float64).reshape(-1)[2])
    return goal_z > ground_z + AIR_GOAL_MARGIN_M


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def evaluate_episodes(model, env, n_episodes: int, base_seed: int) -> dict[str, Any]:
    """用确定性动作跑 ``n_episodes`` 个回合。每次都从 ``base_seed + i`` 重置。"""
    successes: list[float] = []
    holds: list[float] = []
    rewards: list[float] = []
    distances: list[float] = []
    is_air: list[bool] = []
    was_training = model.policy.training
    model.policy.set_training_mode(False)
    started = time.perf_counter()
    try:
        for episode in range(n_episodes):
            obs, _info = env.reset(seed=int(base_seed) + episode)
            air = _goal_is_air(env, obs["desired_goal"])
            done = False
            episode_reward = 0.0
            step_success: list[bool] = []
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(env.action_space.shape)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                step_success.append(_is_success_flag(info.get("is_success", 0.0)))
                done = bool(terminated or truncated)
            if not step_success:
                raise RuntimeError("评估回合没有走任何一步")
            achieved = np.asarray(obs["achieved_goal"], dtype=np.float64).reshape(-1)
            desired = np.asarray(obs["desired_goal"], dtype=np.float64).reshape(-1)
            successes.append(1.0 if step_success[-1] else 0.0)
            hold = len(step_success) >= HOLD_STEPS and all(step_success[-HOLD_STEPS:])
            holds.append(1.0 if hold else 0.0)
            rewards.append(episode_reward)
            distances.append(float(np.linalg.norm(achieved - desired)))
            is_air.append(air)
    finally:
        model.policy.set_training_mode(was_training)
    seconds = time.perf_counter() - started
    ground_success = [flag for flag, air in zip(successes, is_air, strict=True) if not air]
    air_success = [flag for flag, air in zip(successes, is_air, strict=True) if air]
    return {
        "successes": np.asarray(successes, dtype=np.float32),
        "holds": np.asarray(holds, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "final_distances": np.asarray(distances, dtype=np.float32),
        "is_air": np.asarray(is_air, dtype=bool),
        "success_rate": float(np.mean(successes)),
        "hold_rate": float(np.mean(holds)),
        "mean_reward": float(np.mean(rewards)),
        "final_distance": float(np.mean(distances)),
        "success_rate_ground": _mean_or_nan(ground_success),
        "success_rate_air": _mean_or_nan(air_success),
        "n_ground": int(len(ground_success)),
        "n_air": int(len(air_success)),
        "seconds": float(seconds),
    }


def _write_video(path: Path, frames: list[np.ndarray]) -> tuple[int, int]:
    """写成 25 fps 的 mp4。quality 从 5 往下降，直到文件小于 500 KB。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    last_size = -1
    used_quality = -1
    for quality in (5, 2, 0):
        imageio.mimsave(
            path,
            frames,
            fps=VIDEO_FPS,
            format="FFMPEG",
            codec="libx264",
            quality=quality,
            pixelformat="yuv420p",
        )
        last_size = path.stat().st_size
        used_quality = quality
        if last_size < VIDEO_MAX_BYTES:
            return last_size, used_quality
    raise RuntimeError(f"视频仍有 {last_size} 字节，超过 {VIDEO_MAX_BYTES}：{path}")


def record_episode_video(model, env, seed: int, path: Path) -> dict[str, Any]:
    """在单独的 rgb_array 环境上确定性跑 1 个回合并写 mp4。"""
    was_training = model.policy.training
    model.policy.set_training_mode(False)
    started = time.perf_counter()
    try:
        obs, _info = env.reset(seed=int(seed))
        frames = [np.asarray(env.render()).copy()]
        done = False
        step_success: list[bool] = []
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(env.action_space.shape)
            obs, _reward, terminated, truncated, info = env.step(action)
            frames.append(np.asarray(env.render()).copy())
            step_success.append(_is_success_flag(info.get("is_success", 0.0)))
            done = bool(terminated or truncated)
    finally:
        model.policy.set_training_mode(was_training)
    if not step_success:
        raise RuntimeError("视频回合没有走任何一步")
    success = bool(step_success[-1])
    label = "success" if success else "fail"
    final_path = path.parent / f"{path.stem}_{label}{path.suffix}"
    nbytes, quality = _write_video(final_path, frames)
    return {
        "path": final_path,
        "success": success,
        "seconds": float(time.perf_counter() - started),
        "bytes": int(nbytes),
        "quality": int(quality),
        "n_frames": len(frames),
    }


def buffer_transition_count(buffer) -> int:
    """buffer.size() 是时间维上的条数，每条含 n_envs 个环境的转移。"""
    return int(buffer.size()) * int(buffer.n_envs)


class SuccessEvalCallback(BaseCallback):
    """按最后一步成功率评估，并把保持率、距离、贴地/空中分项写入 TensorBoard。"""

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        eval_seed: int,
        run_dir: Path,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.eval_seed = int(eval_seed)
        self.run_dir = Path(run_dir)
        self.best_model_dir = self.run_dir / "best_model"
        self.evaluations_path = self.run_dir / "evaluations.npz"
        self.eval_seconds: list[float] = []
        self.best_success = -1.0
        self.best_success_reward = float("-inf")
        self.best_reward = float("-inf")
        self.best_success_step = -1
        self.best_reward_step = -1

    def _init_callback(self) -> None:
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self._load_selection()

    def _load_selection(self) -> None:
        path = self.best_model_dir / "selection.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.best_success = float(data["best_success_rate"])
        self.best_success_reward = float(data["best_success_mean_reward"])
        self.best_reward = float(data["best_mean_reward"])
        self.best_success_step = int(data.get("best_success_step", -1))
        self.best_reward_step = int(data.get("best_reward_step", -1))

    def _write_selection(self) -> None:
        payload = {
            "best_success_rate": self.best_success,
            "best_success_mean_reward": self.best_success_reward,
            "best_mean_reward": self.best_reward,
            "best_success_step": self.best_success_step,
            "best_reward_step": self.best_reward_step,
        }
        path = self.best_model_dir / "selection.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _append_evaluations(self, step: int, result: dict[str, Any]) -> None:
        row = {
            "timesteps": np.asarray([step], dtype=np.int64),
            "successes": result["successes"][None, :],
            "holds": result["holds"][None, :],
            "rewards": result["rewards"][None, :],
            "final_distances": result["final_distances"][None, :],
            "is_air": result["is_air"][None, :],
            "success_rate": np.asarray([result["success_rate"]], dtype=np.float64),
            "hold_rate": np.asarray([result["hold_rate"]], dtype=np.float64),
            "mean_reward": np.asarray([result["mean_reward"]], dtype=np.float64),
            "final_distance": np.asarray([result["final_distance"]], dtype=np.float64),
            "success_rate_ground": np.asarray([result["success_rate_ground"]], dtype=np.float64),
            "success_rate_air": np.asarray([result["success_rate_air"]], dtype=np.float64),
        }
        if self.evaluations_path.exists():
            with np.load(self.evaluations_path) as old:
                stored = {key: old[key] for key in old.files}
            if stored["successes"].shape[1] != row["successes"].shape[1]:
                raise RuntimeError(
                    "evaluations.npz 的回合数和这次评估不一致，不能追加。"
                    f" 已有 {stored['successes'].shape[1]}，这次 {row['successes'].shape[1]}。"
                )
            merged = {
                key: np.concatenate([stored[key], row[key]], axis=0)
                for key in row
            }
        else:
            merged = row
        np.savez(self.evaluations_path, **merged)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        result = evaluate_episodes(
            self.model,
            self.eval_env,
            self.n_eval_episodes,
            self.eval_seed,
        )
        self.eval_seconds.append(float(result["seconds"]))
        step = int(self.num_timesteps)
        for key in (
            "success_rate",
            "hold_rate",
            "mean_reward",
            "final_distance",
            "success_rate_ground",
            "success_rate_air",
        ):
            self.logger.record(f"eval/{key}", float(result[key]))
        self.logger.dump(step)
        self._append_evaluations(step, result)
        success_rate = float(result["success_rate"])
        mean_reward = float(result["mean_reward"])
        if success_rate > self.best_success + 1e-9 or (
            abs(success_rate - self.best_success) <= 1e-9
            and mean_reward > self.best_success_reward + 1e-9
        ):
            self.best_success = success_rate
            self.best_success_reward = mean_reward
            self.best_success_step = step
            self.model.save(str(self.best_model_dir / "best_success_model"))
            if self.verbose:
                print(
                    f"保存 best_success_model：step={step} success={success_rate:.3f} reward={mean_reward:.3f}",
                    flush=True,
                )
        if mean_reward > self.best_reward + 1e-9:
            self.best_reward = mean_reward
            self.best_reward_step = step
            self.model.save(str(self.best_model_dir / "best_model"))
            if self.verbose:
                print(
                    f"保存 best_model：step={step} reward={mean_reward:.3f}",
                    flush=True,
                )
        self._write_selection()
        print(
            "EVAL "
            f"step={step} success={success_rate:.3f} hold={float(result['hold_rate']):.3f} "
            f"reward={mean_reward:.3f} dist={float(result['final_distance']):.4f} "
            f"ground={float(result['success_rate_ground']):.3f}({result['n_ground']}) "
            f"air={float(result['success_rate_air']):.3f}({result['n_air']}) "
            f"eval_s={float(result['seconds']):.2f}",
            flush=True,
        )
        return True


class VideoEvalCallback(BaseCallback):
    """每次评估后（按 video_every_n_evals）录 1 个回合。"""

    def __init__(
        self,
        video_env,
        eval_freq: int,
        video_every_n_evals: int,
        video_seed: int,
        run_dir: Path,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.video_env = video_env
        self.eval_freq = int(eval_freq)
        self.video_every_n_evals = int(video_every_n_evals)
        self.video_seed = int(video_seed)
        self.video_dir = Path(run_dir) / "videos"
        self.eval_index = 0
        self.video_seconds: list[float] = []
        self.video_bytes: list[int] = []

    def _init_callback(self) -> None:
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        self.eval_index += 1
        if self.video_every_n_evals <= 0:
            return True
        if self.eval_index % self.video_every_n_evals != 0:
            return True
        step = int(self.num_timesteps)
        info = record_episode_video(
            self.model,
            self.video_env,
            self.video_seed,
            self.video_dir / f"step_{step}.mp4",
        )
        self.video_seconds.append(float(info["seconds"]))
        self.video_bytes.append(int(info["bytes"]))
        print(
            f"VIDEO step={step} file={info['path'].name} bytes={info['bytes']} "
            f"quality={info['quality']} frames={info['n_frames']} video_s={info['seconds']:.2f}",
            flush=True,
        )
        return True


class CheckpointLatestCallback(BaseCallback):
    """按 save_freq 保存模型；replay buffer 只保留 checkpoints/replay_buffer_latest.pkl。"""

    def __init__(self, save_freq: int, run_dir: Path, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.checkpoint_dir = Path(run_dir) / "checkpoints"
        self.buffer_path = self.checkpoint_dir / "replay_buffer_latest.pkl"
        self.model_save_seconds: list[float] = []
        self.buffer_save_seconds: list[float] = []

    def _init_callback(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, step: int | None = None) -> None:
        step = int(self.model.num_timesteps if step is None else step)
        model_path = self.checkpoint_dir / f"{step}_steps.zip"
        started = time.perf_counter()
        self.model.save(str(model_path))
        self.model_save_seconds.append(time.perf_counter() - started)
        buffer = getattr(self.model, "replay_buffer", None)
        if buffer is None:
            return
        started = time.perf_counter()
        self.model.save_replay_buffer(str(self.buffer_path))
        self.buffer_save_seconds.append(time.perf_counter() - started)
        if self.verbose:
            print(
                f"CHECKPOINT step={step} model={model_path.name} "
                f"buffer={self.buffer_path.name} transitions={buffer_transition_count(buffer)}",
                flush=True,
            )

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            self.save_checkpoint(int(self.num_timesteps))
        return True


class ThroughputCallback(BaseCallback):
    """分开统计 learning_starts 前后的环境步吞吐量，并采样内存。"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.t_start: float | None = None
        self.steps_start = 0
        self.t_learn: float | None = None
        self.steps_learn: int | None = None
        self.t_end: float | None = None
        self.steps_end: int | None = None
        self.ram_peak_bytes = 0
        self._process = psutil.Process()

    def _on_training_start(self) -> None:
        self.t_start = time.perf_counter()
        self.steps_start = int(self.model.num_timesteps)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._sample_ram()

    def _sample_ram(self) -> None:
        total = 0
        seen: list[psutil.Process] = [self._process]
        try:
            seen.extend(self._process.children(recursive=True))
        except psutil.Error:
            pass
        for proc in seen:
            try:
                total += int(proc.memory_info().rss)
            except psutil.Error:
                continue
        if total > self.ram_peak_bytes:
            self.ram_peak_bytes = total

    def _on_step(self) -> bool:
        if self.n_calls % 50 == 0:
            self._sample_ram()
        # 续训开始时就已经过了 learning_starts，整段都算更新阶段。
        # 新训练则记在第一次跨过阈值的环境步上，这一步本身还没有做梯度更新。
        if self.t_learn is None and int(self.model.num_timesteps) > int(self.model.learning_starts):
            if int(self.steps_start) > int(self.model.learning_starts):
                self.t_learn = self.t_start
                self.steps_learn = int(self.steps_start)
            else:
                self.t_learn = time.perf_counter()
                self.steps_learn = int(self.model.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self.finish()

    def finish(self) -> dict[str, Any]:
        self._sample_ram()
        if self.t_end is None:
            self.t_end = time.perf_counter()
            self.steps_end = int(self.model.num_timesteps)
        assert self.t_start is not None
        assert self.t_end is not None
        assert self.steps_end is not None
        if self.t_learn is None:
            # 整个过程都还没跨过 learning_starts，或者续训开始时就已经跨过了。
            if int(self.steps_start) > int(self.model.learning_starts):
                self.t_learn = self.t_start
                self.steps_learn = self.steps_start
            else:
                self.t_learn = self.t_end
                self.steps_learn = self.steps_end
        assert self.t_learn is not None
        assert self.steps_learn is not None
        pre_steps = int(self.steps_learn - self.steps_start)
        post_steps = int(self.steps_end - self.steps_learn)
        pre_seconds = float(self.t_learn - self.t_start)
        post_seconds = float(self.t_end - self.t_learn)
        gpu_allocated = None
        gpu_reserved = None
        if torch.cuda.is_available():
            gpu_allocated = int(torch.cuda.max_memory_allocated())
            gpu_reserved = int(torch.cuda.max_memory_reserved())
        return {
            "pre_steps": pre_steps,
            "pre_seconds": pre_seconds,
            "pre_sps": (pre_steps / pre_seconds) if pre_seconds > 0 else None,
            "post_steps": post_steps,
            "post_seconds": post_seconds,
            "post_sps_wall": (post_steps / post_seconds) if post_seconds > 0 else None,
            "ram_peak_bytes": int(self.ram_peak_bytes),
            "gpu_peak_allocated_bytes": gpu_allocated,
            "gpu_peak_reserved_bytes": gpu_reserved,
            "learning_starts": int(self.model.learning_starts),
            "num_timesteps_start": int(self.steps_start),
            "num_timesteps_end": int(self.steps_end),
        }
