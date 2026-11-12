"""M0 环境验证：版本、CUDA、MuJoCo、离屏渲染、视频、关键导入、旧环境回归。

路径都从本文件位置推导，不写死盘符。任一检查失败则进程退出码为 1。
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_PATH = ROOT / "requirements.txt"
MODEL_PATH = (ROOT / "models" / "push.xml").resolve()
VIDEO_PATH = ROOT / "runs" / "smoke" / "smoke.mp4"

# (import 名, requirements.txt 发行名)
PACKAGES = (
    ("mujoco", "mujoco"),
    ("gymnasium", "gymnasium"),
    ("gymnasium_robotics", "gymnasium-robotics"),
    ("stable_baselines3", "stable-baselines3"),
    ("sb3_contrib", "sb3-contrib"),
    ("torch", "torch"),
    ("numpy", "numpy"),
    ("imageio", "imageio"),
    ("packaging", "packaging"),
)

_REQ_LINE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*(?P<spec>\S.*?)\s*$"
)

results: list[tuple[str, bool]] = []
model = None
frames: list | None = None


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def emit(line: str) -> None:
    print(line, flush=True)


def record(name: str, ok: bool) -> None:
    results.append((name, ok))


def run_check(name: str, fn) -> None:
    try:
        detail = fn()
    except Exception as exc:
        emit(f"[FAIL] {name}: {exc}")
        traceback.print_exc()
        record(name, False)
        return
    if detail:
        emit(f"[PASS] {name}: {detail}")
    else:
        emit(f"[PASS] {name}")
    record(name, True)


def parse_requirements(text: str) -> dict[str, str]:
    specs: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _REQ_LINE.match(line)
        if match is None:
            continue
        specs[match.group("name").lower()] = match.group("spec").strip()
    return specs


def expected_python(text: str) -> str:
    match = re.search(r"Python\s+(\d+\.\d+)", text)
    if match is None:
        raise RuntimeError("requirements.txt 头部没有 Python 版本说明")
    return match.group(1)


def installed_version(module_name: str, dist_name: str) -> str:
    importlib.import_module(module_name)
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        module = sys.modules[module_name]
        version = getattr(module, "__version__", None)
        if not version:
            raise RuntimeError(f"已导入 {module_name}，但读不到版本")
        return str(version)


def check_python(requirements_text: str) -> str:
    expected = expected_python(requirements_text)
    actual = ".".join(map(str, sys.version_info[:3]))
    if not actual.startswith(expected + ".") and actual != expected:
        raise RuntimeError(f"实际 {actual}，要求 {expected}")
    return f"{actual}（要求 {expected}）"


def check_package(module_name: str, dist_name: str, specs: dict[str, str]) -> str:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    if dist_name not in specs:
        raise RuntimeError(f"requirements.txt 中没有 {dist_name}")
    spec = specs[dist_name]
    version = installed_version(module_name, dist_name)
    if not SpecifierSet(spec).contains(Version(version)):
        raise RuntimeError(f"实际 {version}，要求 {spec}")
    return f"{version}（要求 {spec}）"


def check_cuda() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() 为 False")
    device_name = torch.cuda.get_device_name(0)
    left = torch.randn(4, 4, device="cuda")
    right = torch.randn(4, 4, device="cuda")
    product = left @ right
    torch.cuda.synchronize()
    if not torch.isfinite(product).all():
        raise RuntimeError("GPU 矩阵乘法结果含有非有限值")
    return f"显卡 {device_name}，matmul {tuple(product.shape)}"


def check_mujoco_step() -> str:
    global model
    import mujoco
    import numpy as np

    if not MODEL_PATH.is_file():
        raise RuntimeError("找不到 models/push.xml")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    for _ in range(500):
        mujoco.mj_step(model, data)
    if np.isnan(data.qpos).any() or np.isnan(data.qvel).any():
        raise RuntimeError("qpos 或 qvel 中存在 NaN")
    return f"models/push.xml，mj_step 500 步，qpos={tuple(data.qpos.shape)} qvel={tuple(data.qvel.shape)}"


def check_render() -> str:
    global frames
    import mujoco
    import numpy as np

    if model is None:
        raise RuntimeError("模型未成功加载")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "watching")
    if camera_id < 0:
        raise RuntimeError("模型中没有相机 watching")

    rendered: list = []
    renderer = mujoco.Renderer(model, 240, 320)
    try:
        for index in range(25):
            renderer.update_scene(data, camera="watching")
            frame = np.array(renderer.render(), copy=True)
            if frame.shape != (240, 320, 3):
                raise RuntimeError(f"第 {index} 帧形状为 {frame.shape}，期望 (240, 320, 3)")
            if not np.any(frame):
                raise RuntimeError(f"第 {index} 帧全黑")
            rendered.append(frame)
            if index < 24:
                for _ in range(20):
                    mujoco.mj_step(model, data)
    finally:
        renderer.close()
    frames = rendered
    return "25 帧，形状 (240, 320, 3)，相机 watching，帧间 mj_step 20 次"


def check_video() -> str:
    import imageio.v2 as imageio

    if not frames:
        raise RuntimeError("没有可写入的帧")
    VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        VIDEO_PATH,
        frames,
        fps=25,
        format="FFMPEG",
        codec="libx264",
        pixelformat="yuv420p",
    )
    if not VIDEO_PATH.is_file():
        raise RuntimeError("未生成 mp4 文件")
    size = VIDEO_PATH.stat().st_size
    if size <= 0:
        raise RuntimeError("mp4 文件大小为 0")
    relative = VIDEO_PATH.relative_to(ROOT).as_posix()
    return f"{relative}，{size} bytes"


def check_mujoco_robot_env() -> str:
    from gymnasium_robotics.envs.robot_env import MujocoRobotEnv

    if not isinstance(MujocoRobotEnv, type):
        raise RuntimeError("MujocoRobotEnv 不是类")
    return "gymnasium_robotics.envs.robot_env.MujocoRobotEnv"


def check_tqc() -> str:
    import sb3_contrib

    tqc = getattr(sb3_contrib, "TQC", None)
    if tqc is None:
        raise RuntimeError("sb3_contrib 中没有 TQC")
    return f"sb3_contrib.TQC ({tqc.__module__})"


def check_her() -> str:
    import stable_baselines3

    her = getattr(stable_baselines3, "HerReplayBuffer", None)
    if her is None:
        raise RuntimeError("stable_baselines3 中没有 HerReplayBuffer")
    return f"stable_baselines3.HerReplayBuffer ({her.__module__})"


def check_aloha_push() -> str:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import gymnasium as gym

    import aloha_envs  # noqa: F401  注册 AlohaPush-v0；构造时的 print 是旧环境残留

    env = gym.make("AlohaPush-v0")
    try:
        env.reset()
        for _ in range(10):
            env.step(env.action_space.sample())
    finally:
        env.close()
    return "AlohaPush-v0 reset 后随机动作 10 步"


def main() -> int:
    _configure_stdio()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    requirements_text = REQUIREMENTS_PATH.read_text(encoding="utf-8")
    specs = parse_requirements(requirements_text)

    run_check("版本 Python", lambda: check_python(requirements_text))
    for module_name, dist_name in PACKAGES:
        run_check(
            f"版本 {module_name}",
            lambda module_name=module_name, dist_name=dist_name: check_package(
                module_name, dist_name, specs
            ),
        )
    run_check("CUDA", check_cuda)
    run_check("MuJoCo 加载", check_mujoco_step)
    run_check("离屏渲染", check_render)
    run_check("视频写出", check_video)
    run_check("导入 MujocoRobotEnv", check_mujoco_robot_env)
    run_check("导入 TQC", check_tqc)
    run_check("导入 HerReplayBuffer", check_her)

    emit("--- AlohaPush-v0 构造时的 print 是旧环境残留，属于正常输出 ---")
    run_check("旧环境回归", check_aloha_push)

    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    emit("=" * 48)
    emit(f"汇总: {passed} PASS, {failed} FAIL")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
