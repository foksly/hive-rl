import gym

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

ATARI_OPTS = {
    'policy': 'CnnPolicy',
    'buffer_size': 100000,
    'learning_rate': 1e-4,
    'batch_size': 10000, # 32,
    'learning_starts': 100000,
    'target_update_interval': 1000,
    'train_freq': 64, # 4,
    'gradient_steps': 1,
    'exploration_fraction': 0.1,
    'exploration_final_eps': 0.01,
    'optimize_memory_usage': True,
}

OPTS = {
    'atari': ATARI_OPTS
}

def get_env(name):
    if name == 'atari':
        env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1)
        env = VecFrameStack(env, n_stack=4)
        return env

if __name__ == '__main__':
    env_name = 'atari'
    env = get_env(env_name)

    model_opts = OPTS[env_name]
    model = DQN(**model_opts, env=env, verbose=1, tensorboard_log='/home/foksly/Documents/hive-rl-v2/logs')
    model.learn(total_timesteps=int(5e6), tb_log_name=f"dqn.{env_name}.bs-10k.train_freq-64")
