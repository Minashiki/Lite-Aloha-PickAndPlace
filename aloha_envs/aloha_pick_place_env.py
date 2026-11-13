"""Aloha 左臂 Pick & Place 的 GoalEnv。

只实现 ``MujocoRobotEnv`` 的钩子，不重写 ``step``、``reset`` 和
``_initialize_simulation``。动作是末端增量加夹爪绝对开合，观测布局与 Fetch 的
25 维一致。
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations

# 路径从本文件推导。基类只把以 ``/`` 开头的字符串当绝对路径；这里仍用
# ``os.path.abspath``。Windows 上结果形如 ``D:\...``，基类会把它 ``os.path.join``
# 到包内 assets 后面，而 ``os.path.join`` 遇到带盘符的绝对路径会丢掉前缀，
# 所以最终用的仍是这个绝对路径。
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(_PACKAGE_DIR, "..", "models", "pick_place.xml"))

# 与 scripts/check_model.py、models/pick_place.xml 的 keyframe ``home`` 一致。
HOME_QPOS = np.array(
    [
        4.0629901495e-04,
        -4.7747780177e-02,
        1.0111069065e-01,
        -3.1106394060e-07,
        1.5182284990e00,
        -3.9001127121e-04,
        0.037,
        0.037,
    ],
    dtype=np.float64,
)
GRASP_QUAT = np.array(
    [0.0, -0.7071067811865475, 0.7071067811865476, 0.0],
    dtype=np.float64,
)

# pinch 坐标下的 mocap 裁剪盒，以及物块 / 目标的默认 xy 采样盒。
WS_LOW = np.array([0.15, -0.15, 0.02], dtype=np.float64)
WS_HIGH = np.array([0.40, 0.15, 0.30], dtype=np.float64)
DEFAULT_XY_LOW = np.array([0.25, -0.15], dtype=np.float64)
DEFAULT_XY_HIGH = np.array([0.40, 0.15], dtype=np.float64)

GRIP_CLOSE = 0.002
GRIP_OPEN = 0.037
ACTION_SCALE = 0.05
N_SUBSTEPS = 20
SETTLE_STEPS = 20
_SAMPLE_TRIES = 10_000

ARM_JOINTS = (
    "left/waist",
    "left/shoulder",
    "left/elbow",
    "left/forearm_roll",
    "left/wrist_angle",
    "left/wrist_rotate",
    "left/left_finger",
    "left/right_finger",
)
LEFT_FINGER = "left/left_finger"
RIGHT_FINGER = "left/right_finger"
PINCH_SITE = "left/pinch"
PINCH_BODY = "left/pinch"
OBJ_SITE = "obj_site"
OBJ_JOINT = "obj_joint"
TARGET_SITE = "target"
MOCAP_BODY = "mocap"
GRIPPER_ACTUATOR = "left/gripper"
EVAL_CAMERA = "eval_cam"

for _array in (HOME_QPOS, GRASP_QUAT, WS_LOW, WS_HIGH, DEFAULT_XY_LOW, DEFAULT_XY_HIGH):
    _array.flags.writeable = False


class AlohaPickAndPlaceEnv(MujocoRobotEnv):
    """抓起地面方块并放到随机目标。成功不提前结束，回合长度由 TimeLimit 决定。"""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 25,
    }

    def __init__(
        self,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        p_air: float = 0.5,
        obj_xy_low=DEFAULT_XY_LOW,
        obj_xy_high=DEFAULT_XY_HIGH,
        target_xy_low=DEFAULT_XY_LOW,
        target_xy_high=DEFAULT_XY_HIGH,
        target_height_max: float = 0.20,
        min_obj_grip_dist: float = 0.10,
        min_goal_obj_dist: float = 0.06,
        render_mode: str | None = None,
        width: int = 320,
        height: int = 240,
    ):
        if reward_type not in ("sparse", "dense"):
            raise ValueError(f"reward_type 只能是 sparse 或 dense，收到 {reward_type!r}")
        self.reward_type = reward_type
        self.distance_threshold = float(distance_threshold)
        self.p_air = float(p_air)
        self.obj_xy_low = np.asarray(obj_xy_low, dtype=np.float64).copy()
        self.obj_xy_high = np.asarray(obj_xy_high, dtype=np.float64).copy()
        self.target_xy_low = np.asarray(target_xy_low, dtype=np.float64).copy()
        self.target_xy_high = np.asarray(target_xy_high, dtype=np.float64).copy()
        self.target_height_max = float(target_height_max)
        self.min_obj_grip_dist = float(min_obj_grip_dist)
        self.min_goal_obj_dist = float(min_goal_obj_dist)

        super().__init__(
            model_path=MODEL_PATH,
            initial_qpos=HOME_QPOS,
            n_actions=4,
            n_substeps=N_SUBSTEPS,
            render_mode=render_mode,
            width=width,
            height=height,
        )

        # 基类的 MujocoRenderer 不接收 camera_name，默认不会用 eval_cam。
        self.mujoco_renderer.close()
        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            None,
            width=self.width,
            height=self.height,
            camera_name=EVAL_CAMERA,
        )

    def compute_reward(self, achieved_goal, desired_goal, info):
        """只使用传入的目标。``info`` 可能是 dict、dict 数组或 None。"""
        del info
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == "sparse":
            return -(distance > self.distance_threshold).astype(np.float32)
        return -distance

    def compute_terminated(self, achieved_goal, desired_goal, info):
        del achieved_goal, desired_goal, info
        return False

    def compute_truncated(self, achieved_goal, desired_goal, info):
        del achieved_goal, desired_goal, info
        return False

    def _is_success(self, achieved_goal, desired_goal):
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (distance < self.distance_threshold).astype(np.float32)

    def _cache_ids(self) -> None:
        model = self.model
        names = {
            "pinch_site": (mujoco.mjtObj.mjOBJ_SITE, PINCH_SITE),
            "obj_site": (mujoco.mjtObj.mjOBJ_SITE, OBJ_SITE),
            "target_site": (mujoco.mjtObj.mjOBJ_SITE, TARGET_SITE),
            "pinch_body": (mujoco.mjtObj.mjOBJ_BODY, PINCH_BODY),
            "gripper_act": (mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR),
            "left_finger": (mujoco.mjtObj.mjOBJ_JOINT, LEFT_FINGER),
            "right_finger": (mujoco.mjtObj.mjOBJ_JOINT, RIGHT_FINGER),
        }
        ids = {}
        for key, (obj_type, name) in names.items():
            index = mujoco.mj_name2id(model, obj_type, name)
            if index < 0:
                raise RuntimeError(f"模型里找不到 {name}")
            ids[key] = index
        self._pinch_site_id = ids["pinch_site"]
        self._obj_site_id = ids["obj_site"]
        self._target_site_id = ids["target_site"]
        self._pinch_body_id = ids["pinch_body"]
        self._gripper_act_id = ids["gripper_act"]
        self._left_finger_id = ids["left_finger"]
        self._right_finger_id = ids["right_finger"]

    def _joint_qpos(self, joint_id: int) -> float:
        address = int(self.model.jnt_qposadr[joint_id])
        return float(self.data.qpos[address])

    def _joint_qvel(self, joint_id: int) -> float:
        address = int(self.model.jnt_dofadr[joint_id])
        return float(self.data.qvel[address])

    def _align_mocap_to_pinch(self) -> None:
        """把 mocap 放到 pinch 当前的位置和姿态上。"""
        self._utils.set_mocap_pos(
            self.model, self.data, MOCAP_BODY, self.data.site_xpos[self._pinch_site_id]
        )
        self._utils.set_mocap_quat(
            self.model, self.data, MOCAP_BODY, self.data.xquat[self._pinch_body_id]
        )

    def _write_object(self, xy: np.ndarray) -> None:
        qpos = np.array(
            [xy[0], xy[1], self.initial_object_height, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self._utils.set_joint_qpos(self.model, self.data, OBJ_JOINT, qpos)

    def _sample_object_xy(self) -> np.ndarray:
        grip_xy = self.initial_grip_xpos[:2]
        for _ in range(_SAMPLE_TRIES):
            xy = self.np_random.uniform(self.obj_xy_low, self.obj_xy_high)
            if np.linalg.norm(xy - grip_xy) >= self.min_obj_grip_dist:
                return np.asarray(xy, dtype=np.float64)
        raise RuntimeError("物块 xy 在距离约束下采样失败")

    def _sample_goal(self) -> np.ndarray:
        # 物块已在 ``_reset_sim`` 里放好。空中与否只抽一次，距离不够时只重采位置，
        # 这样接受样本里的空中比例等于 ``p_air``，不会因为空中更容易满足 3D 距离而偏高。
        object_pos = np.array(self.data.site_xpos[self._obj_site_id], dtype=np.float64, copy=True)
        in_air = float(self.np_random.random()) < self.p_air
        ground = self.initial_object_height
        for _ in range(_SAMPLE_TRIES):
            xy = self.np_random.uniform(self.target_xy_low, self.target_xy_high)
            if in_air:
                height = ground + float(self.np_random.uniform(0.0, self.target_height_max))
            else:
                height = ground
            goal = np.array([xy[0], xy[1], height], dtype=np.float64)
            if np.linalg.norm(goal - object_pos) >= self.min_goal_obj_dist:
                return goal
        raise RuntimeError("目标在距离约束下采样失败")

    def _env_setup(self, initial_qpos) -> None:
        self._cache_ids()
        qpos = np.asarray(initial_qpos, dtype=np.float64).reshape(-1)
        if qpos.shape != (len(ARM_JOINTS),):
            raise ValueError(f"initial_qpos 应为 {len(ARM_JOINTS)} 维，实际 {qpos.shape}")
        for name, value in zip(ARM_JOINTS, qpos):
            self._utils.set_joint_qpos(self.model, self.data, name, float(value))
        self._mujoco.mj_forward(self.model, self.data)
        self._align_mocap_to_pinch()
        self.data.ctrl[self._gripper_act_id] = GRIP_OPEN
        for _ in range(SETTLE_STEPS):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # 基类在返回后把 qvel 存成初始状态。清零后，之后每次 reset 的速度都是 0。
        self.data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        self.initial_object_height = float(self.data.site_xpos[self._obj_site_id, 2])
        self.initial_grip_xpos = self.data.site_xpos[self._pinch_site_id].copy()

    def _reset_sim(self) -> bool:
        # ``mj_resetData`` 会把 mocap 还原成 XML 默认值（位置 0.30 0 0.20、单位四元数），
        # 并把 ctrl 清零。夹爪 ctrl 清零后会被截到 0.002，也就是全闭。
        super()._reset_sim()
        self._align_mocap_to_pinch()
        self.data.ctrl[self._gripper_act_id] = GRIP_OPEN
        self._write_object(self._sample_object_xy())
        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _set_action(self, action) -> None:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        pinch = self.data.site_xpos[self._pinch_site_id]
        target = np.clip(pinch + ACTION_SCALE * action[:3], WS_LOW, WS_HIGH)
        self._utils.set_mocap_pos(self.model, self.data, MOCAP_BODY, target)
        self._utils.set_mocap_quat(self.model, self.data, MOCAP_BODY, GRASP_QUAT)
        gripper = GRIP_CLOSE + (action[3] + 1.0) * 0.5 * (GRIP_OPEN - GRIP_CLOSE)
        self.data.ctrl[self._gripper_act_id] = gripper

    def _get_obs(self):
        dt = self.dt
        grip_pos = self.data.site_xpos[self._pinch_site_id].copy()
        object_pos = self.data.site_xpos[self._obj_site_id].copy()
        object_rel_pos = object_pos - grip_pos
        gripper_state = np.array(
            [
                self._joint_qpos(self._left_finger_id),
                self._joint_qpos(self._right_finger_id),
            ],
            dtype=np.float64,
        )
        object_rot = np.ravel(
            rotations.mat2euler(self.data.site_xmat[self._obj_site_id].reshape(3, 3))
        )
        grip_velp = self._utils.get_site_xvelp(self.model, self.data, PINCH_SITE) * dt
        object_velp = self._utils.get_site_xvelp(self.model, self.data, OBJ_SITE) * dt - grip_velp
        object_velr = self._utils.get_site_xvelr(self.model, self.data, OBJ_SITE) * dt
        gripper_vel = (
            np.array(
                [
                    self._joint_qvel(self._left_finger_id),
                    self._joint_qvel(self._right_finger_id),
                ],
                dtype=np.float64,
            )
            * dt
        )
        observation = np.concatenate(
            [
                grip_pos,
                object_pos,
                object_rel_pos,
                gripper_state,
                object_rot,
                np.ravel(object_velp),
                np.ravel(object_velr),
                np.ravel(grip_velp),
                gripper_vel,
            ]
        ).astype(np.float64, copy=False)
        # 构造期基类把 ``self.goal`` 设成空数组后再调用这里。空间形状取的是
        # ``achieved_goal``，所以 ``desired_goal`` 只能原样拷贝，不能做运算。
        return {
            "observation": observation,
            "achieved_goal": object_pos.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _render_callback(self) -> None:
        # ``target`` 直接挂在 worldbody 上，site_pos 就是世界坐标。
        # 构造期 goal 还是空数组，这时渲染不会被调用；空着就跳过。
        if self.goal.shape != (3,):
            return
        self.model.site_pos[self._target_site_id] = self.goal
        self._mujoco.mj_forward(self.model, self.data)

    def _step_callback(self) -> None:
        # ``mj_step`` 之后位置量对应积分前的状态；不刷新的话，渲染会改变轨迹。
        self._mujoco.mj_forward(self.model, self.data)
