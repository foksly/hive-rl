import gym

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

HALFCHEETAH_OPTS = {
    'policy': 'MlpPolicy',
    'batch_size': 10000,  # 64
    'n_steps': 10000,  # 512
    'gamma': 0.98,
    'learning_rate': 2.0633e-05,
    'ent_coef': 0.000401762,
    'clip_range': 0.1,
    'n_epochs': 20,
    'gae_lambda': 0.92,
    'max_grad_norm': 0.8,
    'vf_coef': 0.58096,
    'policy_kwargs': dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=torch.nn.ReLU,
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )
}

ATARI_OPTS = {
    'policy': 'CnnPolicy',
    'batch_size': 256, # 256,
    'n_steps': 128,
    'n_epochs': 4,
    'learning_rate': 2.5e-4,
    'clip_range': 0.1,
    'vf_coef': 0.5,
    'ent_coef': 0.01
}

OPTS = {
    'halfcheetah': HALFCHEETAH_OPTS,
    'atari': ATARI_OPTS
}

def get_env(name):
    if name == 'halfcheetah':
        env = make_vec_env('HalfCheetah-v3', n_envs=1)
        env = VecNormalize(env)
        return env
    elif name == 'atari':
        env = make_atari_env('BreakoutDeterministic-v4', n_envs=8)
        env = VecFrameStack(env, n_stack=4)
        return env

if __name__ == '__main__':
    env_name = 'atari'
    env = get_env(env_name)

    model_opts = OPTS[env_name]
    model = PPO(**model_opts, env=env, verbose=1, tensorboard_log='/home/foksly/Documents/hive-rl-v2/logs')
    model.learn(total_timesteps=int(5e6), tb_log_name=f"ppo.{env_name}_deterministic.zoo_defaults")
