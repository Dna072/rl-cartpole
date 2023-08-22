import argparse

import gymnasium as gym
from gymnasium.wrappers import atari_preprocessing as atp
import torch

import config
from utils import preprocess
from evaluate import evaluate_policy_pong
from dqn import DQN, DCQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['ALE/Pong-v5'], default='ALE/Pong-v5')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'ALE/Pong-v5': config.ALE_Pong_v5,    
}

epoch = 0

# data = torch.load(f'data/returns-epoch-{epoch}.pt')
# print(data)

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env = atp.AtariPreprocessing(env, screen_size=84, 
                                 grayscale_obs=True, frame_skip=1,
                                 noop_max=30, scale_obs=True)
    env_config = ENV_CONFIGS[args.env]


    # Initialize deep Q-networks.
    dqn = DCQN(env_config=env_config).to(device)
    #dqn = torch.load(f'models/ALE_Pong-v5_best.pt', map_location=torch.device('cpu'))

    
    # TODO: Create and initialize target Q-network.
    target_dqn = DCQN(env_config).to(device)
    target_dqn.load_state_dict(dqn.state_dict())

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    # Keep track of steps done
    steps = 0
    returns = []
    OBS_STACK_SIZE = env_config['obs_stack_size']
    for episode in range(env_config['n_episodes']):
        terminated = False
        obs, info = env.reset()

        obs = preprocess(obs, env=args.env).unsqueeze(0)
        obs_stack = torch.cat(OBS_STACK_SIZE *[obs]).unsqueeze(0).to(device)
        
        while not terminated:
            steps += 1
            # Get action from DQN.
            action = dqn.act(obs_stack, env)
            #print('action', action)
            # Act in the true environment.
            action_item = action.item()
            
            next_obs, reward, terminated, truncated, info = env.step(action_item)
            print('action', action_item, action.item(), 'reward', reward)
            reward = torch.tensor([reward], device=device)
            # Preprocess incoming observation.
            if not terminated:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
                next_obs_stack = torch.cat((obs_stack[:, 1:, ...], next_obs.unsqueeze(1)), dim=1).to(device)
            else:
                next_obs = None
                next_obs_stack = None
            
            # Add the transition to the replay memory. Remember to convert
            # everything to PyTorch tensors!
            memory.push(obs_stack, action, next_obs_stack, reward)
            obs_stack = next_obs_stack

            # Run DQN.optimize() every env_config["train_frequency"] steps.
            if steps % env_config["train_frequency"] == 0:
                optimize(dqn, target_dqn, memory, optimizer, device)
            # Update the target network every env_config["target_update_frequency"] steps.
            if steps % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy_pong(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')
            #returns[episode] = mean_return
            returns.append((episode, mean_return))

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/ALE_Pong-v5_best_optimized_nn.pt')
    
    data = {
      'returns': returns,
      'args': env_config
    }
    torch.save(data, f'data/pong-returns-epoch-{epoch}.pt')
    # Close environment after training is completed.
    env.close()

    # Plot returns
    # import matplotlib.pyplot as plt

