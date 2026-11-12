import gymnasium as gym       # ⬅ 必须用 gymnasium
import aloha_envs             # ⬅ 必须 import，让注册代码运行一次

env = gym.make("AlohaPush-v0", render_mode="human")

obs, _ = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
