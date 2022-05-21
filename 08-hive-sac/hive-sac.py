import argparse
import gym
import hivemind
import torch
from stable_baselines3 import HiveSAC
from stable_baselines3.common.env_util import make_vec_env
from torch_optimizer import Lamb
from typing import Callable

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--target-batch-size', type=int, default=10000)
    parser.add_argument('--learning-starts', type=int, default=10000)
    parser.add_argument('--train-freq', type=int, default=1)
    parser.add_argument('--ent-coef', default='auto')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lamb', action='store_true')
    parser.add_argument('--experiment-prefix')
    parser.add_argument('--logdir', default='/home/jheuristic/exp/decentralized/foksly/logs')
    parser.add_argument('--disable-tb', action='store_true')
    parser.add_argument('--n-envs', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--target-update', type=int, default=1)
    parser.add_argument('--initial-peer', type=str)
    args = parser.parse_args()
    return args

def generate_experiment_name(args):
    exp_name_dict = {
        'bs': args.batch_size,
        'target_bs': args.target_batch_size,
        'learning_starts': args.learning_starts,
        'train_freq': args.train_freq,
        'ent_coef': args.ent_coef,
        'n_envs': args.n_envs,
        'buf_size': args.buffer_size,
        'target_upd': args.target_update,
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

def configure_dht_opts(args):
    opts = {
        'start': True,
    }
    if args.initial_peer is not None:
        opts['initial_peers'] = [args.initial_peer]
    return opts

if __name__ == '__main__':
    args = parse_args()

    dht_opts = configure_dht_opts(args)
    print(f'DHT_OPTS: {dht_opts}')
    dht = hivemind.DHT(**dht_opts)
    print("To join the training, use initial_peers =", [str(addr) for addr in dht.get_visible_maddrs()])

    env = make_vec_env('HalfCheetah-v3', n_envs=args.n_envs)
    model_opts = {
        'policy': 'MlpPolicy',
        'env': env,
        'batch_size': args.batch_size,
        'learning_starts': args.learning_starts,
        'train_freq': args.train_freq,
        'ent_coef': args.ent_coef if args.ent_coef == 'auto' else float(args.ent_coef),
        'verbose': 1,
        'tensorboard_log': args.logdir,
        'buffer_size': args.buffer_size,
        'target_update_interval': args.target_update,
        'policy_kwargs': {'share_features_extractor': False}
    }
    if args.lamb:
        model_opts.update(make_lamb_opts(args.lr))
    print(model_opts)

    model = HiveSAC(**model_opts)
    optimizer_params_groups = [
        *model.actor.optimizer.param_groups,
        *model.critic.optimizer.param_groups,
    ]
    if args.ent_coef == 'auto':
        optimizer_params_groups.extend(model.ent_coef_optimizer.param_groups)
    optimizer = torch.optim.Adam(optimizer_params_groups)
    model.hivemind_optimizer = hivemind.Optimizer(
        dht=dht,
        optimizer=optimizer,
        run_id='sac_hivemind',
        batch_size_per_step=args.batch_size,
        target_batch_size=args.target_batch_size,
        offload_optimizer=False,
        verbose=True,
        use_local_updates=False,
        matchmaking_time=3,
        averaging_timeout=10,
    )
    model.hivemind_optimizer.load_state_from_peers()
    model.learn(total_timesteps=int(5e10), tb_log_name=generate_experiment_name(args))
