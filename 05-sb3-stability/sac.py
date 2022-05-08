import argparse
from multiprocessing.dummy import current_process
import gym
import hivemind
from typing import Callable

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from torch_optimizer import Lamb

HALFCHEETAH_OPTS = {
    'policy': 'MlpPolicy',
    'batch_size': 20000,
    'learning_starts': 20000,
    # 'train_freq': 39,
    # 'learning_rate': 1e-3,
}

RACING_OPTS = {
    'policy': 'CnnPolicy',
    'learning_rate': 7.3e-4,
    'buffer_size': 50000,
    'batch_size': 256,
    'ent_coef': 'auto',
    'gamma': 0.98,
    'tau': 0.2,
    'train_freq': 64,
    'gradient_steps': 64,
    'learning_starts': 1000,
    'use_sde': True,
    'use_sde_at_warmup': True,
    'policy_kwargs': {'log_std_init': -2, 'net_arch': [64, 64]}
}

OPTS = {
    'halfcheetah': HALFCHEETAH_OPTS,
    'car_racing': RACING_OPTS,
}

def schedule(learning_rate: float, warmup_rate: float=0.1) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        current_progress = (1 - progress_remaining)
        if current_progress <= warmup_rate:
            return max(learning_rate * (current_progress / warmup_rate), 1e-6)
        return learning_rate
    return func

def get_env(name):
    if name == 'halfcheetah':
        return make_vec_env('HalfCheetah-v3', n_envs=1)
    else:
        env = make_vec_env('CarRacing-v0', n_envs=1)
        env = gym.wrappers.resize_observation.ResizeObservation(env, shape=64)
        env = gym.wrappers.gray_scale_observation.GrayScaleObservation(env, keep_dim=True)
        env = VecFrameStack(env, n_stack=4)
        return env

if __name__ == '__main__':
    env_name = 'halfcheetah'
    env = get_env(env_name)

    model_opts = OPTS[env_name]

    lamb_optimizer_opts = {
        'optimizer_class': Lamb,
        'optimizer_kwargs': {
            'clamp_value': 1e6,
            'debias': True,
            'betas': (0.9, 0.95),
        }
    }
    model = SAC(
        **model_opts,
        learning_rate=schedule(1e-3, warmup_rate=0.1),
        env=env, verbose=1, tensorboard_log='/home/foksly/Documents/hive-rl-v2/logs',
        policy_kwargs=lamb_optimizer_opts,
    )
    model.learn(total_timesteps=int(5e6), tb_log_name="sac.halfcheetah.lamb_optimizer.bs-20k.learning_starts-20k")
