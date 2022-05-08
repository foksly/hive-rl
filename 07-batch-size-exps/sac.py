import argparse
import gym
from typing import Callable

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from torch_optimizer import Lamb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-starts', type=int, default=10000)
    parser.add_argument('--train-freq', type=int, default=1)
    parser.add_argument('--ent-coef', default='auto')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lamb', action='store_true')
    parser.add_argument('--experiment-prefix')
    parser.add_argument('--logdir', default='/home/foksly/Documents/hive-rl-v2/logs/sac-batch-exps')
    parser.add_argument('--disable-tb', action='store_true')
    args = parser.parse_args()
    return args

def generate_experiment_name(args):
    exp_name_dict = {
        'bs': args.batch_size,
        'learning_starts': args.learning_starts,
        'train_freq': args.train_freq,
        'ent_coef': args.ent_coef,
    }

    exp_name = [f'{key}-{value}' for key, value in exp_name_dict.items()]
    exp_name = '.'.join(exp_name)
    if args.lamb:
        exp_name = f'{exp_name}.lamb'

    if args.experiment_prefix:
        exp_name = f'{args.experiment_prefix}.{exp_name}'
    exp_name = exp_name.replace('000.', 'k.')
    return exp_name

def warmup_schedule(learning_rate: float, warmup_rate: float=0.1) -> Callable[[float], float]:
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

def make_lamb_opts(learning_rate, warmup_rate=0.1):
    return {
        'policy_kwargs': {
            'optimizer_class': Lamb,
            'optimizer_kwargs': {
                'clamp_value': 1e6,
                'debias': True,
                'betas': (0.9, 0.95),
            }
        },
        'learning_rate': warmup_schedule(learning_rate, warmup_rate=warmup_rate)
    }

if __name__ == '__main__':
    args = parse_args()

    env = make_vec_env('HalfCheetah-v3', n_envs=1)

    model_opts = {
        'policy': 'MlpPolicy',
        'env': env,
        'batch_size': args.batch_size,
        'learning_starts': args.learning_starts,
        'train_freq': args.train_freq,
        'ent_coef': args.ent_coef if args.ent_coef == 'auto' else float(args.ent_coef),
        'verbose': 1,
        'tensorboard_log': args.logdir
    }
    if args.lamb:
        model_opts.update(make_lamb_opts(args.lr))
    print(model_opts)

    model = SAC(**model_opts)
    model.learn(total_timesteps=int(5e6), tb_log_name=generate_experiment_name(args))
