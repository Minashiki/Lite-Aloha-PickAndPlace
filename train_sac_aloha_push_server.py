import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch
import aloha_envs  # 你的自定义环境包

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 使用设备: {device}")

# ==== 训练参数 ====
TOTAL_STEPS = 5_000_000
EVAL_FREQ = 10_000
EVAL_EPISODES = 10
ENV_ID = "AlohaPush-v0"
LOG_DIR = "./logs_sac_push_parallel"
os.makedirs(LOG_DIR, exist_ok=True)

# ----------- 创建单个环境的函数 -----------
def make_env(rank, render_mode=None):
    def _init():
        env = gym.make(ENV_ID, render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init
if __name__ == "__main__":
    # ----------- 创建并行训练环境 -----------
    NUM_ENVS = 12   # 🟢 推荐：服务器 16 vCPU → 8 并行环境
    train_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])

    # ----------- 单环境评估环境（不能并行）-----------
    eval_env = make_env(999)()

    # SAC 模型
    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=3e-4,
        batch_size=8192, 
        gamma=0.95,
        tau=0.02,
        buffer_size=3_000_000,
        train_freq=1,
        gradient_steps=8,
        ent_coef="auto",
        learning_starts=10_000,
        policy_kwargs=dict(
            net_arch=[2048,2048,2048],
            n_critics=2
        ),
        action_noise=NormalActionNoise(
            mean=np.zeros(6),
            sigma=0.2 * np.ones(6)
        ),
        verbose=1,
        device=device
    )


    # 回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
    )

    # ---------- 开始训练 ----------
    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback)
    model.save(os.path.join(LOG_DIR, "sac_aloha_push_final"))

    train_env.close()
    eval_env.close()
