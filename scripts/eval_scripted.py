"""评估脚本策略：成功率、保持率、步数预算、失败归类和视频。

在项目根目录运行。回合 i 使用 ``seed + i`` 调用 reset。总成功率低于 90% 时退出码为 1。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import aloha_envs  # noqa: F401  注册 AlohaPickAndPlaceSparse-v0
import gymnasium as gym

from scripted_policy import ScriptedPickPlacePolicy

FAIL_UNALIGNED = "未对准就下降"
FAIL_EMPTY = "空抓"
FAIL_DROP = "抬升或搬运中掉落"
FAIL_TIME = "50步内没到目标（时间不够）"
FAIL_LAST = "到了目标但最后一步不在阈值内"
FAIL_OTHER = "其他"

FIELDNAMES = [
    "seed",
    "in_air",
    "obj_x",
    "obj_y",
    "obj_z",
    "goal_x",
    "goal_y",
    "goal_z",
    "success",
    "hold",
    "final_dist",
    "min_dist",
    "first_success_step",
    "t_approach",
    "t_descend",
    "t_grasp",
    "t_lift",
    "t_move",
    "t_hold",
    "n_drop",
    "n_empty",
    "n_timeout",
    "n_press",
    "max_descend_xy",
    "failure",
    "events",
    "transitions",
]


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def _success_flag(value) -> bool:
    return float(np.asarray(value).reshape(-1)[0]) > 0.5


def _distance(obs) -> float:
    return float(np.linalg.norm(np.asarray(obs["achieved_goal"]) - np.asarray(obs["desired_goal"])))


def _make_env(p_air: float, render: bool):
    kwargs = {"p_air": float(p_air)}
    if render:
        kwargs["render_mode"] = "rgb_array"
        kwargs["width"] = 320
        kwargs["height"] = 240
    return gym.make("AlohaPickAndPlaceSparse-v0", **kwargs)


def _annotate(frame: np.ndarray, text: str) -> np.ndarray:
    image = np.ascontiguousarray(frame)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return image
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    width = 6 + 7 * len(text)
    draw.rectangle((0, 0, width, 14), fill=(0, 0, 0))
    draw.text((3, 2), text, fill=(255, 255, 0))
    return np.asarray(canvas, dtype=np.uint8)


def _classify(row_flags: dict) -> str:
    if row_flags["success"]:
        return ""
    if row_flags["descended_unaligned"]:
        return FAIL_UNALIGNED
    if row_flags["n_empty"] > 0 and row_flags["t_lift"] is None:
        return FAIL_EMPTY
    if row_flags["n_drop"] > 0:
        return FAIL_DROP
    if row_flags["ever_success"]:
        return FAIL_LAST
    progressed = (
        row_flags["t_lift"] is not None
        or row_flags["t_move"] is not None
        or row_flags["min_dist"] < 0.15
    )
    if progressed:
        return FAIL_TIME
    if row_flags["n_empty"] > 0:
        return FAIL_EMPTY
    return FAIL_OTHER


def _step_cell(value) -> str:
    if value is None:
        return ""
    return str(int(value))


def run_episode(env, policy: ScriptedPickPlacePolicy, seed: int, render: bool = False):
    obs, _info = env.reset(seed=int(seed))
    policy.reset()
    obj0 = np.asarray(obs["observation"][3:6], dtype=np.float64).copy()
    goal0 = np.asarray(obs["desired_goal"], dtype=np.float64).reshape(3).copy()
    in_air = bool(float(goal0[2]) > float(obj0[2]) + 1e-4)
    successes: list[bool] = []
    distances: list[float] = []
    frames: list[np.ndarray] = []
    if render:
        frames.append(_annotate(env.render(), f"{policy.label} reset"))
    terminated = False
    truncated = False
    for _t in range(50):
        action = policy(obs)
        label = policy.label
        obs, _reward, terminated, truncated, info = env.step(action)
        successes.append(_success_flag(info["is_success"]))
        distances.append(_distance(obs))
        if render:
            frames.append(_annotate(env.render(), f"{label} t={policy._step - 1}"))
        if terminated or truncated:
            break
    final_dist = distances[-1] if distances else _distance(obs)
    min_dist = min(distances) if distances else final_dist
    ever_success = any(successes)
    first_success = next((i for i, flag in enumerate(successes) if flag), None)
    success = bool(successes[-1]) if successes else False
    hold = len(successes) >= 10 and all(successes[-10:])
    flags = {
        "success": success,
        "descended_unaligned": bool(policy.descended_unaligned),
        "n_empty": int(policy.n_empty),
        "n_drop": int(policy.n_drop),
        "t_lift": policy.first_entry["LIFT"],
        "t_move": policy.first_entry["MOVE"],
        "min_dist": float(min_dist),
        "ever_success": ever_success,
    }
    failure = _classify(flags)
    row = {
        "seed": int(seed),
        "in_air": int(in_air),
        "obj_x": float(obj0[0]),
        "obj_y": float(obj0[1]),
        "obj_z": float(obj0[2]),
        "goal_x": float(goal0[0]),
        "goal_y": float(goal0[1]),
        "goal_z": float(goal0[2]),
        "success": int(success),
        "hold": int(hold),
        "final_dist": float(final_dist),
        "min_dist": float(min_dist),
        "first_success_step": _step_cell(first_success),
        "t_approach": _step_cell(policy.first_entry["APPROACH"]),
        "t_descend": _step_cell(policy.first_entry["DESCEND"]),
        "t_grasp": _step_cell(policy.first_entry["GRASP"]),
        "t_lift": _step_cell(policy.first_entry["LIFT"]),
        "t_move": _step_cell(policy.first_entry["MOVE"]),
        "t_hold": _step_cell(policy.first_entry["HOLD"]),
        "n_drop": int(policy.n_drop),
        "n_empty": int(policy.n_empty),
        "n_timeout": int(policy.n_timeout),
        "n_press": int(policy.n_press),
        "max_descend_xy": float(policy.max_descend_xy),
        "failure": failure,
        "events": ";".join(policy.events),
        "transitions": ">".join(f"{name}@{step}" for step, name in policy.transitions),
    }
    return row, frames


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("obj_x", "obj_y", "obj_z", "goal_x", "goal_y", "goal_z", "final_dist", "min_dist", "max_descend_xy"):
                out[key] = f"{float(row[key]):.6f}"
            writer.writerow(out)


def _rate(flags: list[bool]) -> tuple[int, int, float]:
    total = len(flags)
    hit = sum(1 for flag in flags if flag)
    ratio = float(hit) / float(total) if total else 0.0
    return hit, total, ratio


def _fmt_rate(hit: int, total: int, ratio: float) -> str:
    return f"{hit}/{total} = {ratio * 100:.1f}%"


def _select_videos(rows: list[dict], n_videos: int) -> list[dict]:
    if n_videos <= 0 or not rows:
        return []
    picked: list[dict] = []
    used: set[int] = set()

    def take(row: dict) -> bool:
        seed = int(row["seed"])
        if seed in used:
            return False
        used.add(seed)
        picked.append(row)
        return True

    for want_air in (0, 1):
        for row in rows:
            if int(row["in_air"]) == want_air and int(row["success"]) == 1:
                take(row)
                break
    for row in rows:
        if len(picked) >= n_videos:
            break
        take(row)
    for want_air in (0, 1):
        if any(int(item["in_air"]) == want_air for item in picked):
            continue
        for row in rows:
            if int(row["in_air"]) == want_air and take(row):
                break
    if any(int(row["success"]) == 0 for row in rows) and all(int(item["success"]) == 1 for item in picked):
        for row in rows:
            if int(row["success"]) == 0 and take(row):
                break
    return picked


def _save_video(path: Path, frames: list[np.ndarray]) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        path,
        frames,
        fps=25,
        format="FFMPEG",
        codec="libx264",
        pixelformat="yuv420p",
    )


def _video_name(row: dict) -> str:
    kind = "air" if int(row["in_air"]) else "ground"
    outcome = "success" if int(row["success"]) else "fail"
    return f"ep{int(row['seed'])}_{kind}_{outcome}.mp4"


def _hold_steps(rows: list[dict]) -> list[int]:
    steps = []
    for row in rows:
        if int(row["success"]) != 1 or row["t_hold"] == "":
            continue
        steps.append(int(row["t_hold"]))
    return steps


def format_summary(rows: list[dict], args, video_paths: list[Path]) -> str:
    lines: list[str] = []
    lines.append(f"episodes: {len(rows)}")
    if rows:
        lines.append(f"seed: {int(rows[0]['seed'])} .. {int(rows[-1]['seed'])}")
    lines.append(f"p_air: {args.p_air}")
    lines.append("步号: 0 起始的动作序号。进入 HOLD 时，观测里物块距目标已 < 1 cm。")

    def subgroup(pred):
        return [row for row in rows if pred(row)]

    groups = (
        ("总", rows),
        ("贴地", subgroup(lambda row: int(row["in_air"]) == 0)),
        ("空中", subgroup(lambda row: int(row["in_air"]) == 1)),
    )
    lines.append("成功率（最后一步 is_success）:")
    for name, group in groups:
        hit, total, ratio = _rate([int(row["success"]) == 1 for row in group])
        lines.append(f"  {name}: {_fmt_rate(hit, total, ratio)}")
    lines.append("保持率（最后 10 步全部成功）:")
    for name, group in groups:
        hit, total, ratio = _rate([int(row["hold"]) == 1 for row in group])
        lines.append(f"  {name}: {_fmt_rate(hit, total, ratio)}")
    if rows:
        mean_dist = float(np.mean([float(row["final_dist"]) for row in rows]))
        lines.append(f"平均最终距离: {mean_dist:.4f} m")
    else:
        lines.append("平均最终距离: n/a")

    steps = _hold_steps(rows)
    if steps:
        arr = np.asarray(steps, dtype=np.float64)
        lines.append(
            "成功回合到达 HOLD 的步号: "
            f"min={int(np.min(arr))} "
            f"median={float(np.median(arr)):.1f} "
            f"p95={float(np.percentile(arr, 95)):.1f} "
            f"max={int(np.max(arr))} "
            f"n={len(steps)}"
        )
    else:
        lines.append("成功回合到达 HOLD 的步号: n/a")

    counts: dict[str, int] = {}
    for row in rows:
        if int(row["success"]) == 1:
            continue
        counts[row["failure"]] = counts.get(row["failure"], 0) + 1
    lines.append("失败类别:")
    if not counts:
        lines.append("  无")
    else:
        for name in (FAIL_UNALIGNED, FAIL_EMPTY, FAIL_DROP, FAIL_TIME, FAIL_LAST, FAIL_OTHER):
            if name in counts:
                lines.append(f"  {name}: {counts[name]}")
        for name, count in counts.items():
            if name not in (FAIL_UNALIGNED, FAIL_EMPTY, FAIL_DROP, FAIL_TIME, FAIL_LAST, FAIL_OTHER):
                lines.append(f"  {name}: {count}")
    for row in rows:
        if int(row["success"]) == 1:
            continue
        lines.append(
            "FAIL "
            f"seed={int(row['seed'])} "
            f"{'air' if int(row['in_air']) else 'ground'} "
            f"class={row['failure']} "
            f"final={float(row['final_dist']):.4f} "
            f"min={float(row['min_dist']):.4f} "
            f"hold_step={row['t_hold'] or '-'} "
            f"drop={int(row['n_drop'])} empty={int(row['n_empty'])} "
            f"timeout={int(row['n_timeout'])} press={int(row['n_press'])} "
            f"max_descend_xy={float(row['max_descend_xy']):.4f} "
            f"events={row['events'] or '-'} "
            f"trace={row['transitions']}"
        )
    lines.append("视频:")
    if not video_paths:
        lines.append("  无")
    else:
        for path in video_paths:
            size = path.stat().st_size if path.is_file() else 0
            lines.append(f"  {path.as_posix()}  {size} bytes")
    return "\n".join(lines)


def evaluate(args) -> int:
    out = Path(args.out)
    policy = ScriptedPickPlacePolicy()
    env = _make_env(args.p_air, render=False)
    rows: list[dict] = []
    try:
        for index in range(int(args.episodes)):
            row, _frames = run_episode(env, policy, int(args.seed) + index, render=False)
            rows.append(row)
    finally:
        env.close()

    _write_csv(out / "episodes.csv", rows)
    selected = _select_videos(rows, int(args.videos))
    video_paths: list[Path] = []
    if selected:
        render_env = _make_env(args.p_air, render=True)
        try:
            by_seed = {int(row["seed"]): row for row in rows}
            for row in selected:
                _rerun, frames = run_episode(render_env, policy, int(row["seed"]), render=True)
                logged = by_seed[int(row["seed"])]
                if int(_rerun["success"]) != int(logged["success"]) or abs(
                    float(_rerun["final_dist"]) - float(logged["final_dist"])
                ) > 1e-6:
                    print(
                        "WARN 录像回放与统计不一致 "
                        f"seed={int(row['seed'])} "
                        f"logged_success={logged['success']} rerun_success={_rerun['success']} "
                        f"logged_dist={float(logged['final_dist']):.6f} "
                        f"rerun_dist={float(_rerun['final_dist']):.6f}",
                        flush=True,
                    )
                path = out / _video_name(logged)
                _save_video(path, frames)
                video_paths.append(path)
                print(f"video {path.as_posix()} frames={len(frames)}", flush=True)
        finally:
            render_env.close()

    text = format_summary(rows, args, video_paths)
    print(text, flush=True)
    successes = sum(int(row["success"]) for row in rows)
    episodes = len(rows)
    if episodes == 0:
        return 1
    return 0 if successes * 10 >= episodes * 9 else 1


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="评估 Aloha 脚本抓取策略")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p-air", dest="p_air", type=float, default=0.5)
    parser.add_argument("--videos", type=int, default=3)
    parser.add_argument("--out", type=str, default="runs/m3")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = parse_args(argv)
    if args.episodes < 0:
        raise SystemExit("--episodes 不能为负")
    if args.videos < 0:
        raise SystemExit("--videos 不能为负")
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
