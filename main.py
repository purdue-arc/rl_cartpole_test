import gymnasium as gym
import gymnasium_env

from stable_baselines3 import PPO, A2C

models_dir = r"C:\Users\devya\Desktop\Devyansh\Purdue\Clubs\ARC\rl_cartpole_test\models"
TIMESTEPS = 10_000

env = gym.make("gymnasium_env/TouchBall-v0", render_mode="rgb_array")

model = A2C("MultiInputPolicy", env, verbose=1, device="cuda")
model.learn(total_timesteps=TIMESTEPS)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    if done:
      obs = vec_env.reset()

model.save(f"{models_dir}/{TIMESTEPS}")
vec_env.close()