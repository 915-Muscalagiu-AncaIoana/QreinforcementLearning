import gym
import numpy as np
from gym.spaces import Discrete, Box
from display.display_protein import show_protein_pretty

class ProteinFoldingEnvironment(gym.Env):
    def __init__(self, protein_sequence, protein_length):
        self.action_space = Discrete(6)
        self.observation_space = Box(low=np.array([-protein_length, -protein_length]),
                                     high=np.array([protein_length, protein_length]), dtype=np.float32)
        self.protein_length = protein_length
        self.protein_sequence = protein_sequence
        self.current_index = 0
        self.state = (0, 0, 0)
        self.path = [self.state]

    def step(self, action):

        initial_x = self.state[0]
        initial_y = self.state[1]
        initial_z = self.state[2]

        initial_x += self.get_action_meaning(action)[0]
        initial_y += self.get_action_meaning(action)[1]
        initial_z += self.get_action_meaning(action)[2]
        self.state = (initial_x, initial_y, initial_z)
        reward = 0.1
        done = False
        if self.state in self.path:
            reward = 0.01
        elif self.current_index == self.protein_length - 1:
            reward = self.calculate_energy()
            done = True
        self.path.append(self.state)
        self.current_index += 1
        info = {}
        return self.state, reward, done, info

    def calculate_energy(self):
        total_energy = 0
        for index in range(0, self.protein_length-1):
            for jndex in range(0, self.protein_length-1):
                if abs(index - jndex) >= 2:
                    current_amino_acid_i = self.protein_sequence[index]
                    current_amino_acid_j = self.protein_sequence[jndex]
                    current_place_i = self.path[index]
                    current_place_j = self.path[jndex]
                    x_i = current_place_i[0]
                    y_i = current_place_i[1]
                    z_i = current_place_i[2]
                    x_j = current_place_j[0]
                    y_j = current_place_j[1]
                    z_j = current_place_j[2]
                    if current_amino_acid_i == 'H' and current_amino_acid_j == 'H' and (abs(x_i-x_j)+abs(y_i-y_j)+abs(z_i-z_j)== 1):
                        total_energy+=-1
        return total_energy

    def get_action_meaning(self, action):
        return self.get_action_meanings().get(action)

    def get_action_meanings(self):
        return {0: (-1, 0, 0), 1: (1, 0, 0), 2: (0, -1, 0), 3: (0, 1, 0), 4: (0, 0, -1), 5: (0, 0, 1)}

    def render(self):
        X = [space[0] for space in self.path]
        Y = [space[1] for space in self.path]
        Z = [space[2] for space in self.path]
        show_protein_pretty(X,Y,Z)

    def reset(self):
        self.state = (0,0,0)
        self.path = []
        self.current_index = 0
        return self.state
