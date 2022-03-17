import torch
import hivemind
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("CartPole-v1", n_envs=4)

# Setup hivemind optimizer
def hivemind_optimizer_wrapper(parameters, lr, **kwargs):
    kwargs['optimizer'] = lambda params: torch.optim.Adam(params, lr=lr)
    return hivemind.Optimizer(params=parameters, **kwargs)


dht = hivemind.DHT(start=True)
print("To join the training, use initial_peers =", [str(addr) for addr in dht.get_visible_maddrs()])

batch_size = 64
policy_kwargs = {
    'optimizer_class': hivemind_optimizer_wrapper,
    'optimizer_kwargs': {
        'dht': dht,
        'run_id': 'ppo_hivemind',
        'batch_size_per_step': batch_size,
        'target_batch_size': 1024,
        'use_local_updates': False,
        'verbose': False,
    }
}

model = PPO("MlpPolicy", env, verbose=1, batch_size=batch_size, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=250000)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
