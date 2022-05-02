import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Parallel environments
env = make_atari_env('BreakoutDeterministic-v4', n_envs=2)
env = VecFrameStack(env, n_stack=4)

model = PPO(
    'CnnPolicy', env,
    verbose=1, batch_size=64,
    n_steps=512, learning_rate=3e-4,
    tensorboard_log='/home/foksly/Documents/hive-rl-v2/logs'
)
model.learn(total_timesteps=int(5e6), tb_log_name="baseline.bs-64k.envs-2.rout-512k")
