import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
import os
import aloha_envs  # 注册 AlohaPush 环境
from mujoco_viewer import MujocoViewer


# ==== key wrapper：把 truncated 也当成 done ====
class RenderCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.viewer = None

    def _on_training_start(self):
        env0 = self.model.get_env().envs[0]
        self.viewer = MujocoViewer(env0.env.model, env0.env.data)

    def _on_step(self):
        self.viewer.render()
        return True

class TerminateOnTruncatedWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if truncated:
            terminated = True
        return obs, reward, terminated, truncated, info


ENV_ID = "AlohaPush-v0"
reward_type = "sparse"
TOTAL_STEPS = 500_000
LOG_DIR = "./logs_sac_push_render"
os.makedirs(LOG_DIR, exist_ok=True)

# 设置 max_episode_steps（增加步数，避免 batch_size 不匹配）
max_episode_steps = 1000  # 增加最大步数
learning_starts = 1000  # 设置学习开始的时间，确保在采样之前有足够的经验

# ==== 创建训练环境（DummyVecEnv + TerminateOnTruncatedWrapper） ====
train_env = DummyVecEnv([
    lambda: TerminateOnTruncatedWrapper(
        gym.make(ENV_ID, reward_type=reward_type, max_episode_steps=max_episode_steps)
    )
])

# ==== 评估环境 ====
eval_env = DummyVecEnv([
    lambda: Monitor(
        TerminateOnTruncatedWrapper(
            gym.make(ENV_ID, reward_type=reward_type, max_episode_steps=max_episode_steps)
        ),
        filename="./eval_logs"
    )
])

# HER replay buffer 参数
replay_buffer_kwargs = dict(
    n_sampled_goal=4,
    goal_selection_strategy="future",
)

# ==== 使用 SAC 算法进行训练 ====
model = SAC(
    "MultiInputPolicy",
    train_env,
    learning_rate=1e-3,
    buffer_size=1_000_000,
    batch_size=512,
    policy_kwargs=dict(net_arch=[256, 256, 256]),
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=replay_buffer_kwargs,
    tau=0.05,
    gamma=0.95,
    verbose=1,
    ent_coef="auto",
    learning_starts=learning_starts,  # 设置学习开始时采样步数
    tensorboard_log=os.path.join(LOG_DIR, "tb"),
)

# ==== 设置评估回调 ====
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    eval_freq=2000,
    n_eval_episodes=15,
    deterministic=True,
    render=False,
)

# ==== 渲染回调（通过 viewer 实时显示） ====
render_callback = RenderCallback()

# ==== 启动训练 ====
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[eval_callback, render_callback],
    tb_log_name="sac_aloha_push",
)


# ==== 保存最终模型 ====
model.save("sac_her_aloha_push_final")

# ==== 训练结束，清理环境 ====
train_env.close()
eval_env.close()
