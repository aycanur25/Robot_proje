import gym
import json
import numpy as np
import os
import os.path as osp
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime

from DDPG import Actor, Critic
from OU_Noise import OU_Noise
from Replay_Buffer import Replay_Buffer
from utils import model_deep_copy

# """ Global Parameters """
# Environment
env_name = "MountainCarContinuous-v0"

# GPU settings
use_cuda = False
gpu_id = 1

# Replay buffer settings
buffer_size = 1000000
batch_size = 256

# Random seeds
mem_seed = 17
ou_seed = 41

# Learning rates
lr_critic = 2e-2
lr_actor = 3e-3

# Episodes and training parameters
episodes = 450
update_every_n_steps = 20
learning_updates_per_learning_session = 10

# Noise parameters
mu = 0.0
theta = 0.15
sigma = 0.25

# Discount factor
gamma = 0.99

# Gradient clipping
clamp_critic = 5
clamp_actor = 5

# Soft update parameters
tau_critic = 5e-3
tau_actor = 5e-3

# Rolling score settings
win = 100
score_th = 90

# Output path for results
out_path = "results"
if not osp.exists(out_path):
    os.makedirs(out_path)


def update_learning_rate(starting_lr, optimizer, rolling_score_list, score_th):
    """Lowers the learning rate according to how close we are to the solution"""
    if len(rolling_score_list) > 0:
        last_rolling_score = rolling_score_list[-1]
        if last_rolling_score > 0.75 * score_th:
            new_lr = starting_lr / 100.0
        elif last_rolling_score > 0.6 * score_th:
            new_lr = starting_lr / 20.0
        elif last_rolling_score > 0.5 * score_th:
            new_lr = starting_lr / 10.0
        elif last_rolling_score > 0.25 * score_th:
            new_lr = starting_lr / 2.0
        else:
            new_lr = starting_lr
        for g in optimizer.param_groups:
            g['lr'] = new_lr


def run(env, actor_local, actor_target, critic_local, critic_target, optim_actor, optim_critic, memory, ou_noise, device):
    global_step_idx = 0
    score_list = []
    rolling_score_list = []
    max_score = float('-inf')
    max_rolling_score = float('-inf')

    for i_episode in range(episodes):
        start = time.time()
        state_numpy, _ = env.reset()  # New Gym API: env.reset() returns (state, info)
        next_state_numpy = None
        action_numpy = None
        reward = None
        done = False
        score = 0

        while not done:
            # Process current state
            state = torch.from_numpy(state_numpy).float().unsqueeze(0).to(device)
            actor_local.eval()
            with torch.no_grad():
                action_numpy = actor_local(state).cpu().data.numpy().squeeze(0)
            actor_local.train()

            # Add noise to action
            action_numpy += ou_noise.sample()

            # Perform action
            next_state_numpy, reward, terminated, truncated, _ = env.step(action_numpy)  # New Gym API
            done = terminated or truncated
            score += reward

            # Save experience in replay buffer
            memory.add_experience(state_numpy, action_numpy, reward, next_state_numpy, done)

            # Update state
            state_numpy = next_state_numpy

            # Perform training updates
            if len(memory) > batch_size and global_step_idx % update_every_n_steps == 0:
                for _ in range(learning_updates_per_learning_session):
                    states_numpy, actions_numpy, rewards_numpy, next_states_numpy, dones_numpy = memory.sample()
                    states = torch.from_numpy(states_numpy).float().to(device)
                    actions = torch.from_numpy(actions_numpy).float().to(device)
                    rewards = torch.from_numpy(rewards_numpy).float().to(device)
                    next_states = torch.from_numpy(next_states_numpy).float().to(device)
                    dones = torch.from_numpy(dones_numpy).float().unsqueeze(1).to(device)

                    # Critic update
                    with torch.no_grad():
                        next_actions = actor_target(next_states)
                        next_value = critic_target(next_states, next_actions)
                        value_target = rewards + gamma * next_value * (1.0 - dones)
                    value = critic_local(states, actions)
                    loss_critic = F.mse_loss(value, value_target)
                    optim_critic.zero_grad()
                    loss_critic.backward()
                    if clamp_critic is not None:
                        torch.nn.utils.clip_grad_norm_(critic_local.parameters(), clamp_critic)
                    optim_critic.step()

                    # Soft update for critic
                    for target_param, local_param in zip(critic_target.parameters(), critic_local.parameters()):
                        target_param.data.copy_(tau_critic * local_param.data + (1.0 - tau_critic) * target_param.data)

                    # Actor update
                    pred_actions = actor_local(states)
                    loss_actor = -critic_local(states, pred_actions).mean()
                    optim_actor.zero_grad()
                    loss_actor.backward()
                    if clamp_actor is not None:
                        torch.nn.utils.clip_grad_norm_(actor_local.parameters(), clamp_actor)
                    optim_actor.step()

                    # Soft update for actor
                    for target_param, local_param in zip(actor_target.parameters(), actor_local.parameters()):
                        target_param.data.copy_(tau_actor * local_param.data + (1.0 - tau_actor) * target_param.data)

            global_step_idx += 1

        # Logging results
        score_list.append(score)
        rolling_score = np.mean(score_list[-1 * win:])
        rolling_score_list.append(rolling_score)
        max_score = max(max_score, score)
        max_rolling_score = max(max_rolling_score, rolling_score)

        end = time.time()
        print(f"[Episode {i_episode:4d}: score: {score:.2f}; rolling score: {rolling_score:.2f}, "
              f"max score: {max_score:.2f}, max rolling score: {max_rolling_score:.2f}, time cost: {end - start:.2f}s]")

    # Save results
    output = {
        "score_list": score_list,
        "rolling_score_list": rolling_score_list,
        "max_score": max_score,
        "max_rolling_score": max_rolling_score
    }
    json_name = osp.join(out_path, "DDPG.json")
    with open(json_name, 'w') as f:
        json.dump(output, f, indent=4)


# Main function
if __name__ == "__main__":
    env = gym.make(env_name)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")

    # Critic networks
    critic_local = Critic(n_states, n_actions).to(device)
    critic_target = Critic(n_states, n_actions).to(device)
    model_deep_copy(critic_local, critic_target)
    optim_critic = optim.Adam(critic_local.parameters(), lr=lr_critic, eps=1e-4)

    # Replay buffer
    memory = Replay_Buffer(buffer_size, batch_size, mem_seed)

    # Actor networks
    actor_local = Actor(n_states).to(device)
    actor_target = Actor(n_states).to(device)
    model_deep_copy(actor_local, actor_target)
    optim_actor = optim.Adam(actor_local.parameters(), lr=lr_actor, eps=1e-4)

    # Ornstein-Uhlenbeck noise
    ou_noise = OU_Noise(size=n_actions, seed=ou_seed, mu=mu, theta=theta, sigma=sigma)
    ou_noise.reset()

    # Run the DDPG algorithm
    run(
        env=env,
        actor_local=actor_local,
        actor_target=actor_target,
        critic_local=critic_local,
        critic_target=critic_target,
        optim_actor=optim_actor,
        optim_critic=optim_critic,
        memory=memory,
        ou_noise=ou_noise,
        device=device
    )
