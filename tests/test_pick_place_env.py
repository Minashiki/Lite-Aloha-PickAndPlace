"""M2 验收：直接用 ``python tests/test_pick_place_env.py`` 运行，不依赖 pytest。"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

import aloha_envs  # noqa: F401  注册环境
from aloha_envs.aloha_pick_place_env import (
    ACTION_SCALE,
    EVAL_CAMERA,
    GRASP_QUAT,
    GRIP_OPEN,
    HOME_QPOS,
    MODEL_PATH,
    WS_HIGH,
    WS_LOW,
    AlohaPickAndPlaceEnv,
)
from scripts.check_model import GRASP_QUAT as CHECK_GRASP_QUAT
from scripts.check_model import HOME_QPOS as CHECK_HOME_QPOS

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RESULTS: list[tuple[str, bool]] = []


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


def run_check(name: str, fn) -> None:
    try:
        detail = fn()
    except Exception as exc:
        emit(f"[FAIL] {name}: {exc}")
        traceback.print_exc()
        RESULTS.append((name, False))
        return
    if detail:
        emit(f"[PASS] {name}: {detail}")
    else:
        emit(f"[PASS] {name}")
    RESULTS.append((name, True))


def quat_angle_deg(q1, q2) -> float:
    a = np.asarray(q1, dtype=np.float64)
    b = np.asarray(q2, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def mocap_pose(env):
    unwrapped = env.unwrapped
    body_id = mujoco.mj_name2id(unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "mocap")
    mocap_id = int(unwrapped.model.body_mocapid[body_id])
    return (
        unwrapped.data.mocap_pos[mocap_id].copy(),
        unwrapped.data.mocap_quat[mocap_id].copy(),
    )


def assert_reset_state(env) -> str:
    unwrapped = env.unwrapped
    pinch = unwrapped.data.site_xpos[unwrapped._pinch_site_id].copy()
    pinch_quat = unwrapped.data.xquat[unwrapped._pinch_body_id].copy()
    pos, quat = mocap_pose(env)
    pos_err = float(np.linalg.norm(pinch - pos))
    ang_err = quat_angle_deg(pinch_quat, quat)
    ctrl = float(unwrapped.data.ctrl[unwrapped._gripper_act_id])
    qvel = float(np.max(np.abs(unwrapped.data.qvel)))
    obj_z = float(unwrapped.data.site_xpos[unwrapped._obj_site_id, 2])
    z_err = abs(obj_z - unwrapped.initial_object_height)
    if pos_err >= 1e-3:
        raise AssertionError(f"mocap 与 pinch 位置差 {pos_err * 1000:.4f} mm")
    if ang_err >= 0.1:
        raise AssertionError(f"姿态差 {ang_err:.4f} deg")
    if ctrl != GRIP_OPEN:
        raise AssertionError(f"夹爪 ctrl={ctrl}，期望 {GRIP_OPEN}")
    if qvel != 0.0:
        raise AssertionError(f"qvel 最大绝对值 {qvel}")
    if z_err > 1e-8:
        raise AssertionError(f"物块 z 差 {z_err}")
    return f"pos {pos_err * 1000:.4f} mm, ang {ang_err:.4f} deg, z_err {z_err:.3e}"


def drive(env, target, grip: float, steps: int) -> None:
    unwrapped = env.unwrapped
    for _ in range(steps):
        pinch = unwrapped.data.site_xpos[unwrapped._pinch_site_id]
        delta = np.clip((target - pinch) / ACTION_SCALE, -1.0, 1.0)
        action = np.array([delta[0], delta[1], delta[2], grip], dtype=np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break


def captured_warnings(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    texts = []
    for item in caught:
        message = ANSI.sub("", str(item.message)).strip()
        texts.append(f"{item.category.__name__}: {message}")
    return texts


def check_env_pair() -> str:
    lines = []
    for reward_type in ("sparse", "dense"):
        env = AlohaPickAndPlaceEnv(reward_type=reward_type)
        try:
            texts = captured_warnings(lambda env=env: check_env(env.unwrapped))
        finally:
            env.close()
        lines.append(f"{reward_type} warnings={len(texts)}")
        for text in texts:
            lines.append(f"  {text}")
    spec_s = gym.spec("AlohaPickAndPlaceSparse-v0")
    spec_d = gym.spec("AlohaPickAndPlaceDense-v0")
    if spec_s.max_episode_steps != 50 or spec_d.max_episode_steps != 50:
        raise AssertionError(
            f"max_episode_steps sparse={spec_s.max_episode_steps} dense={spec_d.max_episode_steps}"
        )
    if spec_s.kwargs.get("reward_type") != "sparse" or spec_d.kwargs.get("reward_type") != "dense":
        raise AssertionError(f"kwargs sparse={spec_s.kwargs} dense={spec_d.kwargs}")
    return " | ".join(lines)


def check_observation() -> str:
    env = gym.make("AlohaPickAndPlaceSparse-v0")
    try:
        obs, _ = env.reset(seed=0)
        for key, shape in (
            ("observation", (25,)),
            ("achieved_goal", (3,)),
            ("desired_goal", (3,)),
        ):
            if obs[key].shape != shape or obs[key].dtype != np.float64:
                raise AssertionError(f"{key} shape={obs[key].shape} dtype={obs[key].dtype}")
        if not env.observation_space.contains(obs):
            raise AssertionError("reset 观测不在 observation_space 内")
        _match_segments(env, obs, check_velocity=False)
        action = np.array([0.2, -0.4, 0.1, -0.3], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        if not env.observation_space.contains(obs):
            raise AssertionError("step 观测不在 observation_space 内")
        _match_segments(env, obs, check_velocity=True)
    finally:
        env.close()
    return "reset 与一步动作后，各段都与仿真读数一致"


def _match_segments(env, obs, check_velocity: bool) -> None:
    unwrapped = env.unwrapped
    values = obs["observation"]
    grip = unwrapped.data.site_xpos[unwrapped._pinch_site_id].copy()
    obj = unwrapped.data.site_xpos[unwrapped._obj_site_id].copy()
    left = unwrapped.data.qpos[int(unwrapped.model.jnt_qposadr[unwrapped._left_finger_id])]
    right = unwrapped.data.qpos[int(unwrapped.model.jnt_qposadr[unwrapped._right_finger_id])]
    pairs = (
        ("grip_pos", values[0:3], grip),
        ("object_pos", values[3:6], obj),
        ("object_rel_pos", values[6:9], obj - grip),
        ("gripper_state", values[9:11], np.array([left, right], dtype=np.float64)),
    )
    if check_velocity:
        dt = unwrapped.dt
        grip_velp = unwrapped._utils.get_site_xvelp(unwrapped.model, unwrapped.data, "left/pinch") * dt
        object_velp = (
            unwrapped._utils.get_site_xvelp(unwrapped.model, unwrapped.data, "obj_site") * dt
            - grip_velp
        )
        object_velr = unwrapped._utils.get_site_xvelr(unwrapped.model, unwrapped.data, "obj_site") * dt
        from gymnasium_robotics.utils import rotations

        object_rot = np.ravel(
            rotations.mat2euler(unwrapped.data.site_xmat[unwrapped._obj_site_id].reshape(3, 3))
        )
        left_vel = unwrapped.data.qvel[int(unwrapped.model.jnt_dofadr[unwrapped._left_finger_id])]
        right_vel = unwrapped.data.qvel[int(unwrapped.model.jnt_dofadr[unwrapped._right_finger_id])]
        pairs = pairs + (
            ("object_rot", values[11:14], object_rot),
            ("object_velp", values[14:17], object_velp),
            ("object_velr", values[17:20], object_velr),
            ("grip_velp", values[20:23], grip_velp),
            ("gripper_vel", values[23:25], np.array([left_vel, right_vel], dtype=np.float64) * dt),
        )
    for name, got, expected in pairs:
        if not np.allclose(got, expected, rtol=0.0, atol=1e-8):
            raise AssertionError(f"{name} got={got} expected={expected}")
    if not np.allclose(obs["achieved_goal"], obj):
        raise AssertionError("achieved_goal 不是物块位置")
    if not np.allclose(obs["desired_goal"], unwrapped.goal):
        raise AssertionError("desired_goal 不是 self.goal")


def _rollout(env, seed: int, actions):
    obs, _ = env.reset(seed=seed)
    reset_obs = {key: np.copy(value) for key, value in obs.items()}
    steps = []
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        steps.append(
            (
                {key: np.copy(value) for key, value in obs.items()},
                np.float64(reward),
                bool(terminated),
                bool(truncated),
                np.float64(info["is_success"]),
            )
        )
    return reset_obs, steps


def check_reproducible() -> str:
    rng = np.random.default_rng(123)
    actions = rng.uniform(-1.0, 1.0, size=(50, 4)).astype(np.float32)
    env = gym.make("AlohaPickAndPlaceSparse-v0")
    try:
        first_obs, first_steps = _rollout(env, 7, actions)
        second_obs, second_steps = _rollout(env, 7, actions)
        other_obs, _ = _rollout(env, 8, actions[:1])
    finally:
        env.close()
    for key in first_obs:
        if not np.array_equal(first_obs[key], second_obs[key]):
            raise AssertionError(f"同一 seed 的 reset 观测 {key} 不同")
    if len(first_steps) != 50 or len(second_steps) != 50:
        raise AssertionError("动作序列没有走满 50 步")
    for index, (left, right) in enumerate(zip(first_steps, second_steps)):
        for key in left[0]:
            if not np.array_equal(left[0][key], right[0][key]):
                raise AssertionError(f"第 {index} 步 {key} 不同")
        if left[1:] != right[1:]:
            raise AssertionError(f"第 {index} 步 reward/terminated/truncated/success 不同")
        if left[2]:
            raise AssertionError(f"第 {index} 步 terminated=True")
        if left[3] != (index == 49):
            raise AssertionError(f"第 {index} 步 truncated={left[3]}")
    if np.array_equal(first_obs["desired_goal"], other_obs["desired_goal"]) and np.array_equal(
        first_obs["observation"], other_obs["observation"]
    ):
        raise AssertionError("不同 seed 的 reset 结果相同")
    return "seed=7 的 reset 和 50 步完全一致；seed=8 不同；terminated 恒为 False"


def check_reset_repeat() -> str:
    env = gym.make("AlohaPickAndPlaceSparse-v0")
    try:
        env.reset(seed=11)
        env.action_space.seed(11)
        details = [assert_reset_state(env)]
        for index in range(20):
            if index % 2 == 0:
                obj = env.unwrapped.data.site_xpos[env.unwrapped._obj_site_id].copy()
                drive(env, np.array([obj[0], obj[1], 0.10]), 1.0, 10)
                drive(env, np.array([obj[0], obj[1], 0.02]), -1.0, 12)
                drive(env, np.array([obj[0] + 0.05, obj[1], 0.08]), -1.0, 8)
            else:
                for _ in range(12):
                    env.step(env.action_space.sample())
            env.reset()
            details.append(assert_reset_state(env))
    finally:
        env.close()
    return f"首次 reset 加 20 次穿插抓取/随机动作后仍正确；示例 {details[0]}"


def check_render() -> str:
    env = AlohaPickAndPlaceEnv(render_mode="rgb_array")
    try:
        if env.metadata["render_fps"] != 25:
            raise AssertionError(f"render_fps={env.metadata['render_fps']}")
        if int(np.round(1.0 / env.dt)) != 25:
            raise AssertionError(f"1/dt={1.0 / env.dt}")
        env.reset(seed=0)
        env._render_callback()
        site = env.data.site_xpos[env._target_site_id].copy()
        if not np.allclose(site, env.goal, rtol=0.0, atol=0.0):
            raise AssertionError(f"target site {site} != goal {env.goal}")
        camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, EVAL_CAMERA)
        if env.mujoco_renderer.camera_id != camera_id:
            raise AssertionError(
                f"renderer camera_id={env.mujoco_renderer.camera_id}，eval_cam={camera_id}"
            )
        frame = env.render()
        if frame.shape != (240, 320, 3):
            raise AssertionError(f"rgb shape={frame.shape}")
        renderer = mujoco.Renderer(env.model, 240, 320)
        try:
            renderer.update_scene(env.data, camera=EVAL_CAMERA)
            reference = renderer.render()
        finally:
            renderer.close()
        if frame.shape != reference.shape:
            raise AssertionError(f"参考图 shape={reference.shape}")
        diff = np.abs(frame.astype(np.int16) - reference.astype(np.int16))
        mean_diff = float(diff.mean())
        if not np.array_equal(frame, reference) and mean_diff >= 1.0:
            raise AssertionError(f"平均像素差 {mean_diff}")
    finally:
        env.close()
    return f"320x240，camera_id={camera_id}，平均像素差 {mean_diff:.6f}，最大差 {int(diff.max())}"


def _positions_from_kinematics(env):
    """用当前 qpos 和 mocap 在新的 MjData 上做运动学，不改环境自己的 data。"""
    unwrapped = env.unwrapped
    data = mujoco.MjData(unwrapped.model)
    data.qpos[:] = unwrapped.data.qpos
    data.mocap_pos[:] = unwrapped.data.mocap_pos
    data.mocap_quat[:] = unwrapped.data.mocap_quat
    mujoco.mj_kinematics(unwrapped.model, data)
    pinch = data.site_xpos[unwrapped._pinch_site_id].copy()
    obj = data.site_xpos[unwrapped._obj_site_id].copy()
    return pinch, obj


def check_render_trajectory() -> str:
    """同一个 seed、同一组动作：每步渲染与不渲染的 50 步观测必须逐值相同。"""
    actions = np.random.default_rng(5).uniform(-1.0, 1.0, size=(50, 4)).astype(np.float32)

    def rollout(render: bool):
        mode = "rgb_array" if render else None
        env = gym.make("AlohaPickAndPlaceSparse-v0", render_mode=mode)
        try:
            env.reset(seed=3)
            rows = []
            for action in actions:
                obs, _, _, _, _ = env.step(action)
                if render:
                    frame = env.render()
                    if getattr(frame, "shape", None) != (240, 320, 3):
                        raise AssertionError(f"rgb shape={getattr(frame, 'shape', None)}")
                rows.append({key: np.copy(value) for key, value in obs.items()})
            return rows
        finally:
            env.close()

    plain = rollout(False)
    rendered = rollout(True)
    if len(plain) != 50 or len(rendered) != 50:
        raise AssertionError(f"步数 plain={len(plain)} rendered={len(rendered)}")
    max_diff = 0.0
    for index, (left, right) in enumerate(zip(plain, rendered)):
        if set(left) != set(right):
            raise AssertionError(f"第 {index} 步观测键不同")
        for key in left:
            diff = float(np.max(np.abs(left[key] - right[key])))
            max_diff = max(max_diff, diff)
    if max_diff != 0.0:
        raise AssertionError(f"50 步观测最大差 {max_diff:.6e}")
    return "seed=3，50 步 obs 最大差 0"


def check_obs_kinematics() -> str:
    """step 之后的 grip_pos、object_pos 必须等于按当前 qpos/mocap 重算的运动学。"""
    env = gym.make("AlohaPickAndPlaceSparse-v0")
    try:
        max_grip = 0.0
        max_obj = 0.0
        counted = 0
        for episode in range(4):
            env.reset(seed=1 + episode)
            rng = np.random.default_rng(episode)
            for _ in range(50):
                obs, _, _, _, _ = env.step(rng.uniform(-1.0, 1.0, size=4).astype(np.float32))
                pinch, obj = _positions_from_kinematics(env)
                max_grip = max(max_grip, float(np.max(np.abs(obs["observation"][0:3] - pinch))))
                max_obj = max(max_obj, float(np.max(np.abs(obs["observation"][3:6] - obj))))
                counted += 1
    finally:
        env.close()
    if counted < 100:
        raise AssertionError(f"只检查了 {counted} 步")
    if max_grip >= 1e-9 or max_obj >= 1e-9:
        raise AssertionError(
            f"grip_pos 最大绝对差 {max_grip:.3e}，object_pos 最大绝对差 {max_obj:.3e}"
        )
    return (
        f"{counted} 步、4 个回合，grip_pos 最大绝对差 {max_grip:.3e}，"
        f"object_pos 最大绝对差 {max_obj:.3e}"
    )


def check_reward() -> str:
    sparse = AlohaPickAndPlaceEnv(reward_type="sparse")
    dense = AlohaPickAndPlaceEnv(reward_type="dense")
    try:
        rng = np.random.default_rng(0)
        achieved = rng.normal(size=(256, 3))
        desired = rng.normal(size=(256, 3))
        info_array = np.empty(256, dtype=object)
        for index in range(256):
            info_array[index] = {"ignored": index}
        for env in (sparse, dense):
            batch_none = env.compute_reward(achieved, desired, None)
            batch_info = env.compute_reward(achieved, desired, info_array)
            singles = np.stack(
                [
                    np.asarray(env.compute_reward(achieved[i], desired[i], {"k": i}), dtype=np.float64)
                    for i in range(256)
                ]
            )
            if not np.allclose(np.asarray(batch_none, dtype=np.float64), singles, rtol=0.0, atol=1e-8):
                raise AssertionError(f"{env.reward_type} 批量与逐条不一致")
            if not np.allclose(np.asarray(batch_none, dtype=np.float64), np.asarray(batch_info), rtol=0.0, atol=0.0):
                raise AssertionError(f"{env.reward_type} 忽略 info 失败")
            before = np.array(batch_none, copy=True)
            env.reset(seed=1)
            env.step(np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32))
            after = env.compute_reward(achieved, desired, None)
            if not np.array_equal(np.asarray(before), np.asarray(after)):
                raise AssertionError(f"{env.reward_type} 读取了仿真状态")
        distances = np.linalg.norm(achieved - desired, axis=-1)
        sparse_reward = np.asarray(sparse.compute_reward(achieved, desired, None))
        success = np.asarray(sparse._is_success(achieved, desired))
        if not np.array_equal(sparse_reward, -(distances > 0.05).astype(np.float32)):
            raise AssertionError("sparse 奖励公式不对")
        if not set(np.unique(sparse_reward)).issubset({np.float32(0.0), np.float32(-1.0)}):
            raise AssertionError(f"sparse 取值 {np.unique(sparse_reward)}")
        if not np.array_equal(success, (sparse_reward == 0).astype(np.float32)):
            raise AssertionError("随机批量上 success 与 sparse 奖励不一致")
        dense_reward = np.asarray(dense.compute_reward(achieved, desired, None))
        if not np.allclose(dense_reward, -distances, rtol=0.0, atol=1e-8):
            raise AssertionError("dense 奖励不是 -距离")
        for distance, reward_ok, success_ok in (
            (0.05 - 1e-8, np.float32(0.0), np.float32(1.0)),
            (0.05 + 1e-8, np.float32(-1.0), np.float32(0.0)),
            (0.05, np.float32(0.0), np.float32(0.0)),
        ):
            achieved_one = np.zeros(3)
            desired_one = np.array([distance, 0.0, 0.0])
            reward = np.float32(sparse.compute_reward(achieved_one, desired_one, None))
            flag = np.float32(sparse._is_success(achieved_one, desired_one))
            if reward != reward_ok or flag != success_ok:
                raise AssertionError(f"d={distance}: reward={reward} success={flag}")
            dense_one = float(dense.compute_reward(achieved_one, desired_one, None))
            if abs(dense_one + distance) > 1e-12:
                raise AssertionError(f"dense d={distance} reward={dense_one}")
    finally:
        sparse.close()
        dense.close()
    return "N=256 批量一致；d<0.05 成功，d>0.05 失败；d==0.05 时奖励 0 且 success 0（与 Fetch 公式一致）"


def check_random_steps() -> str:
    env = gym.make("AlohaPickAndPlaceSparse-v0")
    try:
        env.reset(seed=0)
        env.action_space.seed(0)
        min_z = np.inf
        max_xy = 0.0
        mocap_low = np.full(3, np.inf)
        mocap_high = np.full(3, -np.inf)
        for _ in range(1000):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            if not np.isfinite(obs["observation"]).all() or not np.isfinite(reward):
                raise AssertionError("出现 NaN 或 Inf")
            if terminated:
                raise AssertionError("随机动作中 terminated=True")
            obj = obs["achieved_goal"]
            min_z = min(min_z, float(obj[2]))
            max_xy = max(max_xy, float(np.hypot(obj[0], obj[1])))
            pos, _ = mocap_pose(env)
            mocap_low = np.minimum(mocap_low, pos)
            mocap_high = np.maximum(mocap_high, pos)
            if np.any(pos < WS_LOW) or np.any(pos > WS_HIGH):
                raise AssertionError(f"mocap 越界 {pos}")
            if float(obj[2]) < 0.015 or float(np.hypot(obj[0], obj[1])) >= 1.0:
                raise AssertionError(f"物块越界 z={obj[2]} xy={np.hypot(obj[0], obj[1])}")
            if truncated:
                env.reset()
    finally:
        env.close()
    return (
        f"min_z={min_z:.6f} max_xy={max_xy:.6f} "
        f"mocap_low={np.array2string(mocap_low, precision=4)} "
        f"mocap_high={np.array2string(mocap_high, precision=4)}"
    )


def _collect_resets(env, n: int, seed: int):
    env.reset(seed=seed)
    rows = []
    for index in range(n):
        if index:
            env.reset()
        unwrapped = env.unwrapped
        obj = unwrapped.data.site_xpos[unwrapped._obj_site_id].copy()
        goal = unwrapped.goal.copy()
        rows.append((obj, goal))
    return rows


def _summarize_distribution(rows, env, p_air: float) -> str:
    unwrapped = env.unwrapped
    objs = np.stack([row[0] for row in rows])
    goals = np.stack([row[1] for row in rows])
    grip_xy = unwrapped.initial_grip_xpos[:2]
    ground = unwrapped.initial_object_height
    obj_dist = np.linalg.norm(objs[:, :2] - grip_xy, axis=1)
    goal_dist = np.linalg.norm(goals - objs, axis=1)
    air = goals[:, 2] > ground + 1e-8
    if np.any(objs[:, 0] < unwrapped.obj_xy_low[0]) or np.any(objs[:, 0] > unwrapped.obj_xy_high[0]):
        raise AssertionError("物块 x 超出采样区")
    if np.any(objs[:, 1] < unwrapped.obj_xy_low[1]) or np.any(objs[:, 1] > unwrapped.obj_xy_high[1]):
        raise AssertionError("物块 y 超出采样区")
    if np.any(goals[:, 0] < unwrapped.target_xy_low[0]) or np.any(goals[:, 0] > unwrapped.target_xy_high[0]):
        raise AssertionError("目标 x 超出采样区")
    if np.any(goals[:, 1] < unwrapped.target_xy_low[1]) or np.any(goals[:, 1] > unwrapped.target_xy_high[1]):
        raise AssertionError("目标 y 超出采样区")
    if np.any(obj_dist < unwrapped.min_obj_grip_dist - 1e-9):
        raise AssertionError(f"物块离夹爪过近 min={obj_dist.min()}")
    if np.any(goal_dist < unwrapped.min_goal_obj_dist - 1e-9):
        raise AssertionError(f"目标离物块过近 min={goal_dist.min()}")
    ratio = float(air.mean())
    if p_air == 0.0 or p_air == 1.0:
        if abs(ratio - p_air) > 0.0:
            raise AssertionError(f"p_air={p_air} 实际比例 {ratio}")
    elif abs(ratio - p_air) > 0.05:
        raise AssertionError(f"空中比例 {ratio} 不在 {p_air}±5% 内")
    if np.any(air):
        air_z = goals[air, 2]
        if np.any(air_z <= ground) or np.any(air_z > ground + unwrapped.target_height_max):
            raise AssertionError(f"空中高度超出 ({air_z.min()}, {air_z.max()}]")
    if np.any(~air):
        if np.max(np.abs(goals[~air, 2] - ground)) > 1e-12:
            raise AssertionError("贴地目标高度不是地面高度")
    air_z_text = "none"
    if np.any(air):
        air_z_text = f"[{goals[air, 2].min():.6f}, {goals[air, 2].max():.6f}]"
    return (
        f"n={len(rows)} p_air={p_air} ratio={ratio:.4f} ground={ground:.6f} "
        f"obj_x=[{objs[:, 0].min():.4f}, {objs[:, 0].max():.4f}] "
        f"obj_y=[{objs[:, 1].min():.4f}, {objs[:, 1].max():.4f}] "
        f"obj_grip_min={obj_dist.min():.4f} "
        f"goal_x=[{goals[:, 0].min():.4f}, {goals[:, 0].max():.4f}] "
        f"goal_y=[{goals[:, 1].min():.4f}, {goals[:, 1].max():.4f}] "
        f"goal_obj_min={goal_dist.min():.4f} air_z={air_z_text}"
    )


def check_distribution() -> str:
    parts = []
    env = AlohaPickAndPlaceEnv(p_air=0.5)
    try:
        parts.append(_summarize_distribution(_collect_resets(env, 1000, 0), env, 0.5))
    finally:
        env.close()
    for p_air in (0.0, 1.0):
        env = AlohaPickAndPlaceEnv(p_air=p_air)
        try:
            parts.append(_summarize_distribution(_collect_resets(env, 200, 0), env, p_air))
        finally:
            env.close()
    return " || ".join(parts)


def check_speed() -> str:
    env = AlohaPickAndPlaceEnv()
    try:
        env.reset(seed=0)
        actions = np.random.default_rng(0).uniform(-1.0, 1.0, size=(1000, 4)).astype(np.float32)
        for action in actions[:20]:
            env.step(action)
        env.reset(seed=0)
        start = time.perf_counter()
        for action in actions:
            env.step(action)
        elapsed = time.perf_counter() - start
    finally:
        env.close()
    rate = 1000.0 / elapsed
    detail = f"{rate:.1f} step/s，用时 {elapsed:.3f} s"
    if rate < 1000.0:
        import cProfile
        import io
        import pstats

        env = AlohaPickAndPlaceEnv()
        try:
            env.reset(seed=0)
            profiler = cProfile.Profile()
            profiler.enable()
            for action in actions[:200]:
                env.step(action)
            profiler.disable()
        finally:
            env.close()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(15)
        emit(stream.getvalue())
        detail += "（低于 1000，见上面的 profile，本项仍记 PASS）"
    return detail


def check_push_regression() -> str:
    env = gym.make("AlohaPush-v0")
    try:
        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action_space.sample())
    finally:
        env.close()
    return "AlohaPush-v0 make、reset、10 步通过（构造时打印的数字是旧环境残留）"


def check_her() -> str:
    from sb3_contrib import TQC
    from stable_baselines3 import HerReplayBuffer

    env = gym.make("AlohaPickAndPlaceSparse-v0")
    try:
        model = TQC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "goal_selection_strategy": "future",
                "n_sampled_goal": 4,
            },
            learning_starts=100,
            batch_size=64,
            device="cpu",
            policy_kwargs={"net_arch": [64, 64]},
            seed=0,
            verbose=0,
        )
        model.learn(total_timesteps=300)
    finally:
        env.close()
    return "TQC+HER learn(total_timesteps=300) 完成"


def check_constants_and_diff() -> str:
    for name in ("step", "reset", "_initialize_simulation"):
        if name in AlohaPickAndPlaceEnv.__dict__:
            raise AssertionError(f"重写了 {name}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        raise AssertionError("找不到 keyframe home")
    qpos_err = float(np.max(np.abs(model.key_qpos[key_id, :8] - HOME_QPOS)))
    quat_err = float(np.max(np.abs(model.key_mquat[key_id].reshape(-1)[:4] - GRASP_QUAT)))
    home_err = float(np.max(np.abs(np.asarray(CHECK_HOME_QPOS) - HOME_QPOS)))
    grasp_err = float(np.max(np.abs(np.asarray(CHECK_GRASP_QUAT) - GRASP_QUAT)))
    if max(qpos_err, quat_err, home_err, grasp_err) >= 1e-8:
        raise AssertionError(
            f"常量误差 key_qpos={qpos_err} key_mquat={quat_err} "
            f"check_home={home_err} check_quat={grasp_err}"
        )
    baseline = subprocess.check_output(
        ["git", "show", "40094cd:aloha_envs/__init__.py"]
    )
    current = (ROOT / "aloha_envs" / "__init__.py").read_bytes()
    if not current.startswith(baseline):
        raise AssertionError("aloha_envs/__init__.py 改动了 40094cd 的原有字节")
    added = current[len(baseline) :]
    if b"AlohaPush" in added:
        raise AssertionError("追加部分包含了 AlohaPush")
    if b"AlohaPickAndPlaceSparse-v0" not in added or b"AlohaPickAndPlaceDense-v0" not in added:
        raise AssertionError("没有追加两个 Pick&Place 注册")
    diff = subprocess.check_output(
        ["git", "diff", "--", "aloha_envs/__init__.py"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    emit("--- git diff aloha_envs/__init__.py ---")
    emit(diff.rstrip("\n"))
    emit("--- end diff ---")
    return (
        f"key_qpos_err={qpos_err:.3e} key_mquat_err={quat_err:.3e} "
        f"check_model 误差 home={home_err:.3e} quat={grasp_err:.3e}；"
        "40094cd 的 __init__.py 字节是当前文件前缀"
    )


def main() -> int:
    _configure_stdio()
    run_check("check_env", check_env_pair)
    run_check("观测", check_observation)
    run_check("可复现", check_reproducible)
    run_check("reset 状态", check_reset_repeat)
    run_check("渲染", check_render)
    run_check("渲染不改变轨迹", check_render_trajectory)
    run_check("观测与运动学一致", check_obs_kinematics)
    run_check("奖励", check_reward)
    run_check("随机 1000 步", check_random_steps)
    run_check("分布", check_distribution)
    run_check("步速", check_speed)
    emit("--- AlohaPush-v0 构造时的 print 是旧环境残留，属于正常输出 ---")
    run_check("AlohaPush 回归", check_push_regression)
    run_check("HER 冒烟", check_her)
    run_check("常量与 diff", check_constants_and_diff)
    passed = sum(1 for _, ok in RESULTS if ok)
    failed = len(RESULTS) - passed
    emit("=" * 48)
    emit(f"汇总: {passed} PASS, {failed} FAIL")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
