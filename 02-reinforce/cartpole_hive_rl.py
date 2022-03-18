from pickletools import optimize
import gym
import hivemind
import torch
import torch.nn as nn
import numpy as np
from agents import CartPoleAgent


DEVICE=torch.device('cuda')
CARTPOLE_N_ACTIONS=2

def generate_session(env, agent, t_max=1000, render=False):
    """ 
    play a full session with REINFORCE agent and train at the session end.
    returns sequences of states, actions andrewards
    """
    # arrays to record session
    states, actions, rewards = [], [], []
    s = env.reset()

    for t in range(t_max):
        if render:
            env.render()
        # action probabilities array aka pi(a|s)
        action_probs = agent.predict_probs(np.array([s]))[0]

        # Sample action with given probabilities.
        a = np.random.choice(env.action_space.n, p=action_probs)
        new_s, r, done, info = env.step(a)

        # record session history to train later
        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = new_s
        if done:
            break

    return states, actions, rewards

def get_cumulative_rewards(rewards, gamma=0.99):
    """
    take a list of immediate rewards r(s,a) for the whole session 
    compute cumulative returns (a.k.a. G(s,a) in Sutton '16)
    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    The simple way to compute cumulative rewards is to iterate from last to first time tick
    and compute G_t = r_t + gamma*G_{t+1} recurrently

    You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
    """
    G = np.zeros(len(rewards))
    G[-1] = rewards[-1]
    for i in range(len(rewards) - 2, -1, -1):
        G[i] = rewards[i] + gamma * G[i + 1]
    return G

def to_one_hot(y_tensor, ndims):
    """ helper: take an integer vector and convert it to 1-hot matrix. """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(
        y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.to(DEVICE)
    return y_one_hot

def compute_loss(logits, actions, rewards, gamma=0.99, entropy_coef=1e-2):
    actions = torch.tensor(actions, dtype=torch.int32, device=DEVICE)
    cumulative_returns = np.array(get_cumulative_rewards(rewards, gamma))
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32, device=DEVICE)

    probs = nn.functional.softmax(logits, -1)
    log_probs = nn.functional.log_softmax(logits, -1)

    log_probs_for_actions = torch.sum(log_probs * to_one_hot(actions, CARTPOLE_N_ACTIONS), dim=1)
    entropy = -torch.mean(torch.sum(probs * log_probs, dim=1))
    loss = -(torch.mean(log_probs_for_actions * cumulative_returns) + entropy_coef * entropy)

    return loss

def train_epoch(env, agent, optimizer):
    states, actions, rewards = generate_session(env, agent)
    states = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)

    optimizer.zero_grad()
    logits = agent(states)
    loss = compute_loss(logits, actions, rewards)
    loss.backward()
    optimizer.step(batch_size=states.shape[0])

    return np.sum(rewards)

def train(n_epochs, env, agent, optimizer, steps_per_epoch=50):
    for _ in range(n_epochs):
        mean_reward = np.mean([train_epoch(env, agent, optimizer) for _ in range(steps_per_epoch)])
        print(f'Mean reward per epoch: {mean_reward:.3f}')


if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    agent = CartPoleAgent()
    agent.to(DEVICE)
    optimizer = torch.optim.Adam(agent.parameters(), 1e-3)

    dht = hivemind.DHT(start=True)
    print("To join the training, use initial_peers =", [str(addr) for addr in dht.get_visible_maddrs()])

    optimizer = hivemind.Optimizer(
        dht=dht,
        run_id='cartpole_reinforce',
        target_batch_size=1000,
        optimizer=optimizer,
        use_local_updates=True,
        matchmaking_time=3.0,
        averaging_timeout=10.0,
        verbose=True
    )
    train(30, env, agent, optimizer)
