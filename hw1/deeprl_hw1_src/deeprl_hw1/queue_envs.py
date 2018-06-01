# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.envs.registration import register
import numpy as np
import np.random.binomial as binomial

class QueueEnv(Env):
    """Implement the Queue environment from problem 3.

    Parameters
    ----------
    p1: float
      Value between [0, 1]. The probability of queue 1 receiving a new item.
    p2: float
      Value between [0, 1]. The probability of queue 2 receiving a new item.
    p3: float
      Value between [0, 1]. The probability of queue 3 receiving a new item.

    Attributes
    ----------
    nS: number of states
    nA: number of actions
    P: environment model
    """
    metadata = {'render.modes': ['human']}

    SWITCH_TO_1 = 0
    SWITCH_TO_2 = 1
    SWITCH_TO_3 = 2
    SERVICE_QUEUE = 3

    def __init__(self, p1, p2, p3):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(
            [(1, 3), (0, 5), (0, 5), (0, 5)])
        self.nS = 3*6*6*6
        self.nA = 4
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def _reset(self):
        """Reset the environment.

        The server should always start on Queue 1.

        Returns
        -------
        (int, int, int, int)
          A tuple representing the current state with meanings
          (current queue, num items in 1, num items in 2, num items in
          3).
        """
        current_queue = 1
        num_1 = 0
        num_2 = 0
        num_3 = 0
        s = tuple([current_queue, num_1, num_2, num_3])
        self.s = s
        self.current_num = [num_1, num_2, num_3]
        return s

    def _step(self, action):
        """Execute the specified action.

        Parameters
        ----------
        action: int
          A number in range [0, 3]. Represents the action.

        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """
        state = self.s
        if action == SERVICE_QUEUE:    
            current_queue = state[0]
        else:
            current_queue = action + 1
        num_1 = max(self.current_num[0] + binomial(n=1, p=self.p1),5)
        num_2 = max(self.current_num[1] + binomial(n=1, p=self.p2),5)
        num_3 = max(self.current_num[2] + binomial(n=1, p=self.p3),5)
        self.current_num = [num_1, num_2, num_3]
        next_state = tuple([current_queue, num_1, num_2, num_3])
        reward = int((action == SERVICE_QUEUE) and (self.current_num[current_queue-1] > 0))        
        is_terminal = False
        self.s = next_state
        return next_state, reward, is_terminal, None

    def _render(self, mode='human', close=False):
        pass

    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        pass

    def query_model(self, state, action):
        """Return the possible transition outcomes for a state-action pair.

        This should be in the same format at the provided environments
        in section 2.

        Parameters
        ----------
        state
          State used in query. Should be in the same format at
          the states returned by reset and step.
        action: int
          The action used in query.
          
        Returns
        -------
        [(prob, nextstate, reward, is_terminal), ...]
          List of possible outcomes
        """
        if action == SERVICE_QUEUE:    
            next_queue = state[0]
        else:
            next_queue = action + 1
        outcome = []
        if state[1] == 5:
            p_q1 = [1.0]
        else:
            p_q1 = [1-self.p1, self.p1]
            
        if state[2] == 5:
            p_q2 = [1.0]
        else:
            p_q2 = [1-self.p2, self.p2]
            
        if state[3] == 5:
            p_q3 = [1.0]
        else:
            p_q3 = [1-self.p3, self.p3]
            
        for a,i in enumerate(p_q1):
            for b,j in enumerate(p_q2):
                for c,k in enumerate(p_q3):
                    prob = i*j*k
                    nextstate = tuple([next_queue, state[1]+a, state[2]+b, state[3]+c])
                    reward = int((action == SERVICE_QUEUE) and (nextstate[nextstate[0]] > 0))
                    outcome.append(tuple([prob, nextstate, reward, False]))
        return outcome

    def get_action_name(self, action):
        if action == QueueEnv.SERVICE_QUEUE:
            return 'SERVICE_QUEUE'
        elif action == QueueEnv.SWITCH_TO_1:
            return 'SWITCH_TO_1'
        elif action == QueueEnv.SWITCH_TO_2:
            return 'SWITCH_TO_2'
        elif action == QueueEnv.SWITCH_TO_3:
            return 'SWITCH_TO_3'
        return 'UNKNOWN'


register(
    id='Queue-1-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .9,
            'p3': .1})

register(
    id='Queue-2-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .1,
            'p3': .1})
