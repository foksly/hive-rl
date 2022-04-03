import gym
import torch
import hivemind
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

BATCH_SIZE=64

dht = hivemind.DHT(start=True)
print("To join the training, use initial_peers =", [str(addr) for addr in dht.get_visible_maddrs()])

env = make_atari_env('BreakoutDeterministic-v4', n_envs=16, seed=0)                                                      
env = VecFrameStack(env, n_stack=4)                          
model = PPO('CnnPolicy', env, verbose=1, batch_size=BATCH_SIZE)
model.policy.optimizer_class = hivemind.Optimizer
model.policy.optimizer = hivemind.Optimizer(
    dht=dht,
    optimizer=model.policy.optimizer,
    run_id='ppo_hivemind',
    batch_size_per_step=BATCH_SIZE,
    target_batch_size=100000,
    use_local_updates=True,
    offload_optimizer=False,
    verbose=True,
)
model.learn(total_timesteps=int(5e6))
