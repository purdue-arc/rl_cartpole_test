from gymnasium.envs.registration import register

register(
    id="gymnasium_env/TouchBall-v0",
    entry_point="gymnasium_env.envs:TouchBallNoPhysicsEnv",
)