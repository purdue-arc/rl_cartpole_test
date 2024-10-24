from gymnasium.envs.registration import register

register(
    id="gymnasium_env/TouchBall-v0",
    entry_point="gymnasium_env.envs:TouchBallNoPhysicsEnv",
    max_episode_steps=300,
    reward_threshold=None,
    kwargs = {"render_mode":"human"}
)