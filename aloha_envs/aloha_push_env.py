import os
import time

MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "../models/", "push.xml")


import mujoco
import numpy as np
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations
from typing import Optional, Any, SupportsFloat

class AlohaPushEnv(MujocoRobotEnv):
    """
    Aloha Push Task（关节空间控制, 6DoF arm）
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    ARM_JOINTS = [
        "left/waist",
        "left/shoulder",
        "left/elbow",
        "left/forearm_roll",
        "left/wrist_angle",
        "left/wrist_rotate",
    ]

    def __init__(
        self,
        model_path: str = MODEL_XML_PATH,
        n_substeps=10,
        reward_type="sparse",
        distance_threshold=0.05,
        **kwargs,
    ):
        self.model_path = model_path
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.max_episode_steps = 2000
        self.current_step = 0

        self.neutral_q = np.array([0.0, -0.96, 1.16, 0.0, -0.30, 0.0])
        self.goal = np.zeros(3)

        super().__init__(
            n_actions=len(self.ARM_JOINTS),
            n_substeps=n_substeps,
            model_path=self.model_path,
            initial_qpos=self.neutral_q,
            **kwargs,
        )

        self.obj_xy_range = 0.3
        self.goal_xy_range = 0.3
        self.goal_x_offset = 0.0
        self.goal_z_range = 0.0

        self.goal_range_low = np.array([-self.goal_xy_range / 2 + self.goal_x_offset, -self.goal_xy_range / 2, 0])
        self.goal_range_high = np.array(
            [self.goal_xy_range / 2 + self.goal_x_offset, self.goal_xy_range / 2, self.goal_z_range])
        self.obj_range_low = np.array([-self.obj_xy_range / 2, -self.obj_xy_range / 2, 0])
        self.obj_range_high = np.array([self.obj_xy_range / 2, self.obj_xy_range / 2, 0])

        # 是把整个区域在 x 方向上平移（桌子位置偏右）
        self.goal_range_low[0] += 0.3
        self.goal_range_high[0] += 0.3
        self.obj_range_low[0] += 0.3
        self.obj_range_high[0] += 0.3

    # ------------------------------------
    # Simulation Init
    # ------------------------------------
    def _initialize_simulation(self):
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.initial_time = self.data.time
        self.initial_qvel = self.data.qvel.copy()

        self._reset_arm()

        # object initial height
        obj_geom_id = self._model_names.geom_name2id["obj_geom"]
        self.initial_object_z = float(self.model.geom_size[obj_geom_id][2])
        print(self.initial_object_z)

        self._mujoco_step()

    def _reset_arm(self):
        for name, q in zip(self.ARM_JOINTS, self.neutral_q):

            self._utils.set_joint_qpos(self.model, self.data, name, q)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------
    # Step
    # ------------------------------------
    def step(self, action):
        # delta-joint control
        scale = 0.02
        action = np.clip(action, -1, 1)


        for i, jname in enumerate(self.ARM_JOINTS):
            cur_q = self._utils.get_joint_qpos(self.model, self.data, jname)
            target_q = cur_q + scale * action[i]

            jid = self._model_names.joint_name2id[jname]
            low, high = self.model.jnt_range[jid]
            target_q = np.clip(target_q, low, high)

            self._utils.set_joint_qpos(self.model, self.data, jname, target_q)

        self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        # reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], obs["ee_pose"], {})
        success = self._is_success(obs["achieved_goal"], obs["desired_goal"])
        terminated = bool(success)
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        info = {"is_success": success, "action": action}

        return obs, reward, terminated, truncated, info

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    # ------------------------------------
    # Observation
    # ------------------------------------
    def _get_obs(self):

        ee_pos = self._utils.get_site_xpos(self.model, self.data, "ee_center_site")

        obj_pos = self._utils.get_site_xpos(self.model, self.data, "obj_site")

        obs = np.concatenate([ee_pos, obj_pos])

        return {
            "observation": obs.copy(),
            "achieved_goal": obj_pos.copy(),
            "desired_goal": self.goal.copy(),
        }

    # ------------------------------------
    # Reset
    # ------------------------------------
    def _reset_sim(self):

        self.data.time = self.initial_time
        self.data.qvel[:] = self.initial_qvel
        self.current_step = 0
        self._reset_arm()
        # self._reset_arm()

        # reset object
        base = np.array([0.55, 0.0, self.initial_object_z])
        noise = self.np_random.uniform([-0.10, -0.15, 0], [0.10, 0.15, 0])
        obj_pos = base + noise
        obj_qpos = np.concatenate([obj_pos, [1, 0, 0, 0]])

        self._utils.set_joint_qpos(self.model, self.data, "obj_joint", obj_qpos)
        self.goal = self._sample_goal()
        # self._sample_goal()
        self._sample_object()
        self._render_callback()
        mujoco.mj_forward(self.model, self.data)


        return True

    # ------------------------------------
    # Goal
    # ------------------------------------
    def _sample_goal(self):
        base = np.array([0.0, 0.0, self.initial_object_z])
        # noise = self.np_random.uniform([-0.12, -0.15, 0], [0.12, 0.15, 0])
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return base + noise

    def _sample_object(self) -> None:
        object_position = np.array([0.0, 0.0, self.initial_object_z])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        object_xpos = np.concatenate([object_position, np.array([1, 0, 0, 0])])
        self._utils.set_joint_qpos(self.model, self.data, "obj_joint", object_xpos)

    def goal_distance(self, goal_a, goal_b) -> SupportsFloat:
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
    def compute_reward(self, achieved_goal, desired_goal, info) -> SupportsFloat:

        d = self.goal_distance(achieved_goal, desired_goal)
        ee_pose = self._utils.get_site_xpos(self.model, self.data, "ee_center_site")
        ee2obj = self.goal_distance(ee_pose, achieved_goal)
        # ee2obj_distance = self.goal_distance(ee_pose, achieved_goal)
        if self.reward_type == "sparse":
            # 稀疏奖励
            # 如果 d > self.distance_threshold，表示目标未达成，返回 负奖励 -1；
            # 如果 d <= self.distance_threshold，表示目标达成，返回 正奖励 0
            return -(d > self.distance_threshold).astype(np.float32)
            # return -(0.3 * ee2obj + d)
        else:
            return -d

    # def _is_success(self, achieved, desired):
    #     return np.linalg.norm(achieved - desired) < self.distance_threshold
    def _is_success(self, achieved, desired):
        diff = achieved - desired
        if diff.ndim == 1:
            d = np.linalg.norm(diff)
            return bool(d < self.distance_threshold)
        else:
            d = np.linalg.norm(diff, axis=-1)
            return (d < self.distance_threshold)

    def _render_callback(self):
        # 把 target site 的位置改到 self.goal
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._model_names.site_name2id["target"]
        self.model.site_pos[site_id] = self.goal - sites_offset[site_id]

        # 重新 forward 生效
        mujoco.mj_forward(self.model, self.data)
