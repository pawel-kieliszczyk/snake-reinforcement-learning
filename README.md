# SnakeAI

Training AI to play a perfect game of Snake. With reinforcement learning (distributed A2C) it can learn to play a perfect game and score maximum points.


## Overview

AI learning to play Snake game "from pixels" with Tensorflow 2.0.

![](snake-animation.gif)


## Requirements

Python 2 and Tensorflow 2.0 Beta or later


## Usage

To train AI, simply type:

```
$ python src/train.py
```

The agent can be trained multiple times. It will keep improving. Its state is saved automatically.

If you want to watch your trained AI playing the game:

```
$ python src/play.py
```

The repository contains a pre-trained AI (trained on 1 GPU + 12 CPUs). To watch it playing, type:

```
$ python src/play_pretrained.py
```

## Implementation details

Implementation uses a distributed version of Advantage Actor-Critic method (A2C).
It consists of two types of processes:
 + **master process** (1 instance): It owns the neural network model. It broadcasts network's weights to all "worker" processes (see below) and waits for mini-batches of experiences. Then it combines all the mini-batches and performs a network update using SGD. Then it broadcasts the current neural network's weights to workers again.
 + **worker process** (as many as number of cores): Each worker has its own copy of an A2C agent. Neural networks weights are received from "master" process (see above). Sample Snake games are played, a mini-batch of experiences is collected and sent back to master. Each worker then waits for an updated set of network's weights.


Neural network architecture:
 + Shared layers by both actor and critic: 4x convlutional layer (filters: 3x3, channels: 64).
 + Actor's head (policy head): 1x convolutional layer (filters: 1x1, channels: 2), followed by a fully connected layer (4 units, one per move: up, down, left, right)
 + Critic's head (value head): 1x convolutional layer (filters: 1x1, channels: 1), followed by a fully connected layer (64 units), followed by a fully connected layer (1 unit - state's value)
