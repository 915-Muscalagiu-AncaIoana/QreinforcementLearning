import numpy as np
from pip._internal.utils.misc import tabulate
import random

class Qagent(object):
    """
    Implementation of a Q-learning Algorithm
    """

    def __init__(
        self,
        action_size: int,
        state_size: int,
        learning_parameters: dict,
        exploration_parameters: dict,
        name: str = "agent",
        color: str = "r",
    ) -> None:
        """ initialize the q-learning agent

        Args:
            action_size (int): number of actions the agent can take
            state_size (int): numbe of states the env has
            learning_parameters (dict): learning paramters of the agent
            exploration_parameters (dict): exploration paramters for the agent
            name (str, optional):  set the name of the Q-Agent. Defaults to "agent".
            color (str, optional): set the color of the agent for plotting. Defaults to "r".
        """
        self.name = name
        self.color = color

        self.action_size = action_size
        self.state_size = state_size
        self.qtable = np.zeros((state_size, action_size))

        self.learning_rate = learning_parameters["learning_rate"]
        self.gamma = learning_parameters["gamma"]

        self.epsilon = exploration_parameters["epsilon"]
        self.max_epsilon = exploration_parameters["max_epsilon"]
        self.min_epsilon = exploration_parameters["min_epsilon"]
        self.decay_rate = exploration_parameters["decay_rate"]

    def update_qtable(
            self, state: int, new_state: int, action: int, reward: int, done: bool
    ) -> None:
        """
        update the q-table: Q(s,a) = Q(s,a) + lr  * [R(s,a) + gamma * max Q(s',a') - Q (s,a)]

        Args:
          state (int): current state of the environment
          new_state (int): new state of the environment
          action (int): current action taken by agent
          reward (int): current reward received from env
          done (boolean): variable indicating if env is done
        """
        new_qvalue = (
                reward
                + self.gamma * np.max(self.qtable[new_state, :]) * (1 - done)
                - self.qtable[state, action]
        )
        self.qtable[state, action] = (
                self.qtable[state, action] + self.learning_rate * new_qvalue
        )

    def update_epsilon(self, episode: int) -> None:
        """
        reduce epsilon, exponential decay

        Args:
          episode (int): number of episode
        """
        self.epsilon = self.min_epsilon + (
                self.max_epsilon - self.min_epsilon
        ) * np.exp(-self.decay_rate * episode)

    def get_action(self, state: int) -> int:
        """
        select action e-greedy.exploration-exploitation trade-off:
        - exploitation, max value for given state
        - exploration, random choice

        Args:
          state (int): current state of the environment/agent

        Returns:
          action (int): action that the agent will take in the next step
        """
        if random.uniform(0, 1) >= self.epsilon:
            action = np.argmax(self.qtable[state, :])
        else:
            action = np.random.choice(self.action_size)
        return action

    def __str__(self, tablefmt="fancy_grid") -> None:
        """plot the q-table. Generate the table in fancy format.
        """
        headers = [f"Action {action}" for action in range(self.action_size)]
        showindex = [f"State {state}" for state in range(self.state_size)]
        table = tabulate(self.qtable, headers=headers, showindex=showindex, tablefmt="fancy_grid")
        return f"{self.name}\n{table}"