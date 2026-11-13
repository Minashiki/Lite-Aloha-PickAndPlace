from gymnasium.envs.registration import register

# Push task
register(
    id="AlohaPush-v0",
    entry_point="aloha_envs.aloha_push_env:AlohaPushEnv",
)

register(
    id="AlohaPickAndPlaceSparse-v0",
    entry_point="aloha_envs.aloha_pick_place_env:AlohaPickAndPlaceEnv",
    kwargs={"reward_type": "sparse"},
    max_episode_steps=50,
)

register(
    id="AlohaPickAndPlaceDense-v0",
    entry_point="aloha_envs.aloha_pick_place_env:AlohaPickAndPlaceEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=50,
)
