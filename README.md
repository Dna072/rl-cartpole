# rl-cartpole and Pong

![image](https://miro.medium.com/v2/resize:fit:704/1*hHXVW-Unl96OHJOZ1J_4ig.gif)

# Introduction

In this report, we discuss the paper *Playing Atari with Deep
Reinforcement Learning*[@mnih2013], implement the algorithm Deep
Q-learning on the gymnasium environments *CartPole-v1* and
*ALE\\Pong-v5*. In both cases we make use of algorithm described in the
paper.

# How Deep Q-learning works

Deep Q-learning is an algorithm which makes use of Q-learning; an
off-policy learning method and a Deep Neural Network for non-linear
function approximation. According to the paper \"Playing Atari with Deep
Reinforcement Learning\" [1], a Deep Q-Network was implemented
to learn to play Atari games simply by taking frames of the current
screen at a particular time, and using a Convolutional Neural Network
(CNN) to approximate a function that when an input frame (frame from
screen) is presented to it, an output of the action to take is provided
by the network.

The training of the network is done with Experience Replay, which is
storing and replaying game states (current state, action, reward, next
state) that the Reinforcement learning algorithm is able to learn from.
Experience Replay can be used in off-policy algorithms for planning (or
offline learning). Off-policy methods are able to update the algorithm's
parameters using saved and stored information from previously taken
actions. With this approach small batches of randomly (according to
uniform distribution) selected samples of experience, i.e (current
state, action, reward, next state) are drawn from a pool of histories
(or memory). [2] The memory has a specific length, this only *N*
last experience can be drawn and not all the experience. Random batches
of experiences are drawn in order to average the behavior distribution
over many of previous states, smoothing out learning and avoid
oscillations or divergence of the parameters. These are the new ideas
that make the algorithm work.

It is also worth mentioning that for playing the atari games, the state
is not simply a frame of the screen at time *t*, but at least 3 frames
is used to represent a state depending on the game since the agent is
supposed to be in motion and just with just a frame, we will not be able
to tell the direction or speed of the agent, and thus multiple frames
are needed to represent a state.

# Deep Q-Network for CartPole-v1 environment

In the Gymnasium CartPole-v1 environment we have a continous state space
of length 4 which represent the cart position, cart velocity, pole angle
and pole angular velocity. We have 2 actions i.e. left and right. We
design Deep Q-network with 4 inputs (representing the states), two
hidden layers and 2 outputs (representing the actions). Each fully
connected layer has ReLU as the activation function.

## Plot of average returns during training for 1000 episodes

![image](https://github.com/Dna072/rl-cartpole/blob/main/plots/returns-episode-1.png?raw=true)

## Discussion

The hyper-parameter choices greatly affect the training. Below is the
list of hyper parameters and how they each affected the training.

-   **Target Network Update Frequency**: It represents the frequency at
    which update the target network to be the same as the current
    network. This affects the stability and speed for convergence. We
    realised low values say 10 - 50, did not work quite well. We stuck
    with the value 100.

-   **Batch Size**: The batch size determines how many experiences are
    sampled from the replay buffer to update the network in each
    training iteration. A larger batch size can lead to more stable
    learning, but it might also slow down training due to increased
    computational requirements. We realised that decreasing the batch
    size reduced our average returns during the training, and a higher
    batch size had almost the same average returns as a the default of
    32 for the project, and was a slower to train.

-   **Replay Buffer Size**: DQN uses a replay buffer to store and sample
    past experiences for training. The size of the replay buffer affects
    how much historical data is used for learning. A larger buffer can
    help stabilize learning, while a smaller buffer might lead to more
    rapid forgetting. Since the batches are drawn from the replay
    buffer, we kept the buffer size large to minimize forgetting by the
    network and ensure stabilized learning. A reduced buffer size, had a
    very slow average rewards improvement rate

-   **Exploration Strategy**: We used $\epsilon$-greedy exploratory
    strategy with exponential decay from 1.0 to 0.05, at an anneal rate
    of $10^4$. This means we started the algorithm by picking random
    actions to gain as much information for all states and actions, and
    gradually reduce the $\epsilon$-greedy to 0.05

# Learning to play Pong

In the Gymnasium ALE\\Pong-v5 environment, we have the observation space
as the frames at each timestep during an episode. This is represented as
Box(0, 255, (210, 160,3), uint8), where the pixel values are \[0-255\]
and 3 channels of 210 by 160 frames. To serve as input to our DQN, we do
not need all the channels for each frame and thus we preprocess the
environment to return grayscale frames.

The action space consists of 6 discrete actions,

-   0 - NOOP (No operation)

-   1 - FIRE (not used)

-   2 - RIGHT

-   3 - LEFT

-   4 - RIGHT_FIRE (same as action 2)

-   5 - LEFT_FIRE (same as action 3)

In effect, we have three distinct actions, i.e. No action, RIGHT and
LEFT. The only actions that cause the agent to move is 2/4 and 3/5 and
we make use of this knowledge to help our network learn faster.

## Policy network architecture

In this problem, our input is image frames, and thus we use
Convolutional Neural Network architecture as described in the Nature DQN
paper. The layers are initialized with these values:

``` {.python language="Python"}
self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
    self.fc1 = nn.Linear(3136, 512)
    self.fc2 = nn.Linear(512, self.n_actions)
```

## Pre-processing

We make use of the AtariPreprocessing module from gymansium.wrappers to
get preprocessed frames scaled to a square image (84 by 84 pixels) and
color format to grayscale so we only have one channel per frame. The
pixels of each frame are also normalized to $[0,1]$ from $[0,255]$.

## Stacking observations

A frame represents matrix of pixels of the current screen. With just one
frame, it is impossible to represent the state of the environment since
we cannot tell the direction in which the opponent or agent is moving
and their velocities. For the Pong environment, an observation size of 4
frames can adequately represent the state of the environment, without
losing any information.

We represent a state as a stack of 4 frames. At the start of each
episode, we initialize the frame stack to have 4 of the same frame and
then push new observation frames at the bottom of the stack while
popping one frame from the stack. This is done as follows

    obs_stack = torch.cat(obs_stack_size * [obs])
                     .unsqueeze(0)
                    .to(device)
    next_obs_stack = torch.cat((obs_stack[:, 1:, ...],
                               obs.unsqueeze(1)), dim=1)
                               .to(device)

## Training the network

The algorithm for training the network works in a similar way compared
to the CartPole-v1 environment with some small tweaks. The network for
the ALE\\Pong-v5 environment is a CNN with 3 convolutional layers and
two linear layers. The ouput from the final convolutional layer is
flattened before its passed to the fully connected layers. Each
observation is tensor of size \[1,4,84,84\] (representing the stacked
frames). The output is vector of size \[$n\_actions$\] i.e. number of
actions.

The actions that actually move the agent are 2/4 (moves agent left) and
3/5 (moves agent right). And thus, having only two outputs (representing
the 2 valid actions) for the network is ideal and should improve the
training time since we don't need to approximate weights for the other
actions which do not have an effect on the agent. However, we could not
train the network with only two outputs, we realised it did not learn
anything new and kept getting the least rewards per episode. We may have
to re-look our implementation of the network with 2 outputs.

A workaround we used to train the network was to focus only on the 2
actions during the exploration period (according to $\epsilon$-greedy)
and then pick the action with maximum Q-value during the greedy period.
This way we increase the Q-values of the valid actions and make sure
network keeps improving those values more than the values for other
actions. The network learned quite well with this approach and we will
be able to see from the plots from the episodic returns.

We trained our network on Google Colab with GPUs. Since the availability
is not guaranteed on the free resources, we saved our DQN when it
provides higher returns compared to the previous DQN, and start each
epoch with the last saved DQN model.


## Plots

![image](https://github.com/Dna072/rl-cartpole/blob/main/plots/pong/pong-dqcn-returns-episode-11.png?raw=true)

![image](https://github.com/Dna072/rl-cartpole/blob/main/plots/pong/pong-dqcn-returns-episode-0.png?raw=true)

![image](https://github.com/Dna072/rl-cartpole/blob/main/plots/pong/pong-dqcn-returns-episode-1.png?raw=true)

At some point we had issues with Google Colab, so the data was not well
gathered for epochs 4 and 5, but our model was updated during these
epoch. Thus we continue with plots for epoch 6 - 10. The extra epochs we
realised were not very necessary to train the network.

![image](https://github.com/Dna072/rl-cartpole/blob/main/plots/pong/pong-dqcn-returns-episode-4.png?raw=true)

![image](https://github.com/Dna072/rl-cartpole/blob/main/plots/pong/pong-dqcn-returns-episode-5.png?raw=true)

![image](https://github.com/Dna072/rl-cartpole/blob/main/plots/pong/pong-dqcn-returns-episode-6.png?raw=true)

## Evaluating the network

We evaluated the network between epochs to verify that the network is
actually learning. For the Pong environment, we also recorded the
frames.
[Download](https://github.com/Dna072/rl-cartpole/blob/main/video/rl-video-episode-e0.mp4)
and view the agent in action after the first epoch, with $\epsilon$ set
to 0.05. The mean return of the policy was 16.0 for one of evaluation.

After the last epoch, we evaluated the policy again this time with
$\epsilon$ set to 0, fully exploiting the learned Q-values. The mean
return of the policy was 21.0 for one episode of evaluation.
[Download](https://github.com/Dna072/rl-cartpole/blob/main/video/rl-video-episode-e-final-1.mp4)
to view the agent with the best policy learnt so far.

We however noticed, not surprisingly, that when the difficulty of the
environment was increased to 3, the best policy learnt so far got a mean
score of -19.0. Thus we trained for one epoch a new network in the
difficulty 3 Pong environment. The mean return at the end of one episode
was 14.0. Training further will improve the policy.
[Download](https://github.com/Dna072/rl-cartpole/blob/main/video/rl-video-episode-df_3-0.mp4)
to view the agent with the policy.

# References
[1] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and
M. Riedmiller, “Playing atari with deep reinforcement learning,” Computing in Science
& Engineering, 2013.

[2] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction. MIT Press,
2018.
