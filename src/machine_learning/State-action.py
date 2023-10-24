import numpy as np
from utilsc3w3ml import *

num_states = 6
num_actions = 2

terminal_left_reward = 100
terminal_right_reward = 10
each_step_reward = 0

# discount
gamma = 0.9

# probability of going in wrong direction
misstep_prob = 0

generate_visualization(terminal_left_reward, terminal_right_reward, each_step_reward, gamma, misstep_prob)
plt.show()
