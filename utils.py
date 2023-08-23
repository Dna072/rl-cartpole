import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v1', 'ALE/Pong-v5']:
        return torch.tensor(obs, device=device).float()
    else:
        raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')

def print_pong_episodic_returns():
    j = 1
    for i in range(12):
        file_path = f'data/pong-returns-epoch-{i}.pt'
        data = torch.load(file_path, map_location=torch.device('cpu'))

        returns = data['returns']
        target_update_frequency = data['args']['target_update_frequency']
        train_frequency = data['args']['train_frequency']
        gamma = data['args']['gamma']
        eps_start = data['args']['eps_start']
        eps_end = data['args']['eps_end']
        n_episodes = data['args']['n_episodes']

        x = [x for x,y in returns]
        y = [y for x,y in returns]

        fig = plt.figure()
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.9)

        # Set titles for the figure and the subplot respectively
        fig.suptitle(f'Epoch {j+1}', fontsize=10, fontweight='bold')
        j+=1
        ax.set_title('Average returns at episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average return')

        plot_x = max(x) - 200
        plot_y = min(y) + 1
        ax.text(plot_x, plot_y, f'target_update_frequency = {target_update_frequency} \n gamma = {gamma} \n eps_start = {eps_start} \n eps_end = {eps_end} \n train_frequency = {train_frequency}',
                bbox={'facecolor': 'orange', 'alpha':0.3, 'pad':5})

        plt.xlim(0, max(x))
        plt.plot(x,y)
        plt.show()
        
        fig.savefig(f'plots/pong/pong-dqcn-returns-episode-{i}.png', format='png')

        print(data)
        

def print_cartpole_episodic_returns():
    for i in range(3):
        file_path = f'data/returns-epoch-{i}.pt'
        data = torch.load(file_path, map_location=torch.device('cpu'))

        print(data)
        returns = data
        # target_update_frequency = data['args']['target_update_frequency']
        # train_frequency = data['args']['train_frequency']
        # gamma = data['args']['gamma']
        # n_episodes = data['args']['n_episodes']

        x = [x for (x,y) in returns]
        y = [y for (x,y)in returns]

        print(x)

        fig = plt.figure()
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.9)

        # Set titles for the figure and the subplot respectively
        fig.suptitle(f'Epoch {i+1}', fontsize=10, fontweight='bold')
        ax.set_title('Average returns at episodes')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average return')

        # ax.text(250, -1, f'target_update_frequency = {target_update_frequency} \n gamma = {gamma} \n train_frequency = {train_frequency}',
        #         bbox={'facecolor': 'orange', 'alpha':0.3, 'pad':5})

        plt.plot(x,y)
        plt.show()
        
        fig.savefig(f'plots/cartpole/returns-episode-{i+1}.png', format='png')
        

        #print(data)
        


if __name__ == '__main__':
    print_pong_episodic_returns()