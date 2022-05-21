import argparse
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-steps', type=int, default=128)
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-epochs', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--experiment-prefix', type=str)
    args = parser.parse_args()
    return args

def generate_experiment_name(args):
    exp_name_dict = {
        'bs': args.batch_size,
        'n_envs': args.n_envs,
        'n_steps': args.n_steps,
        'n_epochs': args.n_epochs,
    }

    exp_name = [f'{key}-{value}' for key, value in exp_name_dict.items()]
    exp_name = '.'.join(exp_name)

    if args.experiment_prefix:
        exp_name = f'{args.experiment_prefix}.{exp_name}'
    exp_name = exp_name.replace('000.', 'k.')
    return exp_name

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

if __name__ == "__main__":
    args = parse_args()

    # Parallel environments
    env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=args.n_envs)
    env = VecFrameStack(env, n_stack=4)

    model = PPO(
        'CnnPolicy', env,
        verbose=1,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        clip_range=0.1,
        vf_coef=0.5,
        ent_coef=0.01,
        tensorboard_log='/home/foksly/misc/logs/on_policy'
    )
    model.learn(total_timesteps=int(1e7), tb_log_name=generate_experiment_name(args))
