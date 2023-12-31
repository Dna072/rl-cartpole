import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('observation', 'action', 'next_observation', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        # return tuple(zip(*sample))
        return sample


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.anneal_step = 0

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def act(self, observation, env, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
        math.exp(-1 * self.anneal_step / self.anneal_length)

        self.anneal_step += 1
        sample = random.random()

        if sample > eps_threshold:
            with torch.no_grad():
                # greedy mode
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self(observation).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

class DCQN(nn.Module):
    def __init__(self, env_config):
        super(DCQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.anneal_step = 0

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

        ## Actions:
        # RIGHT: 2
        # LEFT:  3

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.fc1(self.flatten(x)))
        x = self.fc2(x)

        return x

    def act(self, observation, env, valid_actions=[2,3], exploit=False):
        ## Actions
        # 0 - 2
        # 1 - 3
        """Selects an action with an epsilon-greedy exploration strategy."""
        # Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # Implement epsilon-greedy exploration.
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
        math.exp(-1 * self.anneal_step / self.anneal_length)

        self.anneal_step += 1
        sample = random.random()

        if sample > eps_threshold or exploit == True:
            with torch.no_grad():
                # greedy mode
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self(observation).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.choice(valid_actions)]], device=device, dtype=torch.long)         

def optimize(dqn, target_dqn, memory, optimizer, device):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!
    transitions = memory.sample(dqn.batch_size)
    batch = Transition(*zip(*transitions))

    # a mask to filter final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_observation)),
                                  device=device, dtype=bool)
    non_final_next_states = torch.cat([s for s in batch.next_observation if s is not None])

    state_batch = torch.cat(batch.observation)
    reward_batch = torch.cat(batch.reward)
    action_batch = torch.cat(batch.action)

    #Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    q_values = dqn(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(dqn.batch_size, device=device)

    # Compute the Q-value targets. Only do this for non-terminal transitions!
    with torch.no_grad():
        next_state_values[non_final_mask] = target_dqn(non_final_next_states).max(1)[0]

    # Compute the expected Q values
    q_value_targets = (next_state_values * dqn.gamma) + reward_batch

    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()
