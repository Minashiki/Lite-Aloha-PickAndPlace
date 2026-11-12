from gymnasium.envs.registration import register

# Push task
register(
    id="AlohaPush-v0",
    entry_point="aloha_envs.aloha_push_env:AlohaPushEnv",
)