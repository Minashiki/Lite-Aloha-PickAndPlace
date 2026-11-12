import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise

import aloha_envs
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.her import HerReplayBuffer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 使用设备: {device}")
# ==== 训练参数 ====
TOTAL_STEPS = 500_00000         # 先设短一点便于调试
EVAL_FREQ = 2_000
EVAL_EPISODES = 15
ENV_ID = "AlohaPush-v0"
LOG_DIR = "./logs_sac_push_render"
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(render_mode=None):
    """创建可选渲染环境"""
    env = gym.make(ENV_ID, render_mode=render_mode)
    env = Monitor(env)
    return env

if __name__ == "__main__":
    render_mode = None
    # 训练环境（关闭渲染，提高速度）
    train_env = make_env(render_mode=render_mode)
    # 评估环境（打开渲染）
    eval_env = make_env(render_mode=render_mode)

    # 兼容 SB3==2.2.1 的 HER 参数
    # replay_buffer_kwargs = dict(
    #     n_sampled_goal=4,
    #     goal_selection_strategy="future",
    # )



    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        # replay_buffer_class=HerReplayBuffer,
        # replay_buffer_class=ReplayBuffer,
        # replay_buffer_kwargs=replay_buffer_kwargs,
        learning_rate=1e-3,
        batch_size=512,
        gamma=0.95,
        tau=0.05,
        buffer_size=1_000_000,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto_0.2",
        learning_starts=1_000,
        policy_kwargs=dict(
            net_arch=[256, 256, 256],  # Network size = [256,256,256]
            n_critics=2  # Number of critics = 2
        ),
        action_noise=NormalActionNoise(mean=np.zeros(6), sigma=0.2 * np.ones(6)),  # N(0,0.2)
        verbose=1,
        tensorboard_log=os.path.join(LOG_DIR, "tb"),
        device=device,  # 👈 指定训练设备
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    # === 启动训练 ===
    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback)
    model.save(os.path.join(LOG_DIR, "sac_her_aloha_push_final"))

    train_env.close()
    eval_env.close()
    img = eval_env.render()
    # 例如：cv2.imshow("FrankaPush", img)

    # === 可视化回放 ===
    print("\n🎬 回放训练结果...")
    input("")
    render_env = make_env(render_mode="human")
    model = SAC.load(os.path.join(LOG_DIR, "sac_her_aloha_push_final"), env=render_env)

    obs, _ = render_env.reset()
    for step in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = render_env.step(action)
        render_env.render()
        time.sleep(0.01)
        if done or truncated:
            obs, _ = render_env.reset()
    render_env.close()
