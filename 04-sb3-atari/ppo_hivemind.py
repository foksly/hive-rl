import argparse
import gym
import torch
import hivemind
from transformers.optimization import get_linear_schedule_with_warmup
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--peer-id', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-envs', type=int, default=2)
    parser.add_argument('--n-rollout-steps', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--target-batch-size', type=int, default=10000)    
    parser.add_argument('--no-use-local-updates', dest='use_local_updates', action='store_false')
    parser.add_argument('--experiment-prefix')
    parser.add_argument('--logdir', default='/home/foksly/Documents/hive-rl-v2/logs')
    parser.add_argument('--disable-tb', action='store_true')
    args = parser.parse_args()
    return args


def configure_dht_opts(args):
    dht_opts = {'start': True}
    if args.peer_id > 1:
        dht_opts.update({
            'initial_peers': ['/ip4/127.0.0.1/tcp/36747/p2p/QmbJd26UVFUthazqYhkycVCp9dcPHH2zCLnrdEdsnyXu4y'],
        })
    else:
        dht_opts.update({
            'identity_path': '/home/foksly/Documents/hive-rl-v2/hive-rl.identity', 
            'host_maddrs': ['/ip4/127.0.0.1/tcp/36747'],
        })

    return dht_opts


def configure_hivemind_opts(args):
    hivemind_opts = {
        'run_id': 'ppo_hivemind',
        'batch_size_per_step': args.batch_size,
        'target_batch_size': args.target_batch_size,
        'offload_optimizer': False,
        'verbose': True,
    }

    if args.use_local_updates:
        hivemind_opts.update({
            'use_local_updates': True,
            'delay_state_averaging': True,
        })
    else:
        hivemind_opts.update({
            'use_local_updates': False,
            'delay_state_averaging': False,
            'matchmaking_time': 5,
        })
    return hivemind_opts


def generate_experiment_name(args):
    exp_name_dict = {
        'peer_id': args.peer_id,
        'bs': args.batch_size,
        'target_bs': args.target_batch_size,
        'n_envs': args.n_envs,
        'ro_steps': args.n_rollout_steps,
    }

    exp_name = [f'{key}-{value}' for key, value in exp_name_dict.items()]
    exp_name = ['local_updates' if args.use_local_updates else 'avg_grads'] + exp_name
    exp_name = '.'.join(exp_name)
    if args.experiment_prefix:
        exp_name = f'{args.experiment_prefix}.{exp_name}'
    exp_name = exp_name.replace('000.', 'k.')
    return exp_name


def configure_tb_opts(args):
    if args.disable_tb:
        return {}, {}
    model_init_tb_opts = {'tensorboard_log': args.logdir}
    model_learn_tb_opts = {'tb_log_name': generate_experiment_name(args)}
    return model_init_tb_opts, model_learn_tb_opts


if __name__ == "__main__":
    args = parse_args()
    dht_opts = configure_dht_opts(args)
    dht = hivemind.DHT(**dht_opts)

    env = make_atari_env('BreakoutDeterministic-v4', n_envs=args.n_envs)
    env = VecFrameStack(env, n_stack=4)

    model_init_tb_opts, model_learn_tb_opts = configure_tb_opts(args)
    model = PPO(
        'CnnPolicy', env,
        verbose=1, batch_size=args.batch_size,
        n_steps=args.n_rollout_steps, learning_rate=args.lr,
        **model_init_tb_opts
    )

    scheduler = lambda opt: get_linear_schedule_with_warmup(
        opt, num_warmup_steps=args.warmup_steps, num_training_steps=1000,
    )
    model.policy.optimizer_class = hivemind.Optimizer
    model.policy.optimizer = hivemind.Optimizer(
        dht=dht,
        optimizer=model.policy.optimizer,
        scheduler=scheduler,
        # grad_averager_factory=None,
        **configure_hivemind_opts(args),
    )
    if args.peer_id > 1:
        model.policy.optimizer.load_state_from_peers()
    model.learn(total_timesteps=int(5e6), **model_learn_tb_opts)
