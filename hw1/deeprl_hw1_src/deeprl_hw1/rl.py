# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np


def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """

    # synchronous update
    ns = env.unwrapped.nS
    na = env.unwrapped.nA
    old_value_func = np.zeros(ns)
    new_value_func = np.zeros(ns)
    iter_num = 0
    for _ in range(max_iterations):
        new_value_func = np.zeros(ns)
        iter_num += 1
        delta = 0
        for s in range(ns):
            for (prob, nextstate, reward, is_terminal) in env.unwrapped.P[s][policy[s]]:
                new_value_func[s] += prob*(reward + gamma*old_value_func[nextstate])
            delta = max(delta, abs(new_value_func[s]-old_value_func[s]))
        if delta < tol:
            break
        old_value_func = new_value_func
    return new_value_func, iter_num
    

def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    ns = env.unwrapped.nS
    na = env.unwrapped.nA
    action_value = np.zeros((ns, na), dtype=np.float32)
    for s in range(ns):
        for a in range(na):
            for (prob, nextstate, reward, is_terminal) in env.unwrapped.P[s][a]:
                action_value[s][a] += prob * (reward + gamma * value_function[nextstate])
    greedy_policy = np.argmax(action_value, axis=1)
    return greedy_policy


def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    new_policy = value_function_to_policy(env, gamma, value_func)
    return ~(policy == new_policy).all(), new_policy


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    ns = env.unwrapped.nS
    policy = np.zeros(ns, dtype='int')
    policy_iter_num = 0
    value_iter_num = 0
    for _ in range(max_iterations):
        policy_iter_num += 1
        value_func, num = evaluate_policy(env, gamma, policy, max_iterations, tol)
        value_iter_num += num
        non_stopped, policy = improve_policy(env, gamma, value_func, policy)
        if ~non_stopped:
            break
    value_func, num = evaluate_policy(env, gamma, policy, max_iterations, tol)
    value_iter_num += num
    return policy, value_func, policy_iter_num, value_iter_num
    

def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    ns = env.unwrapped.nS
    na = env.unwrapped.nA
    old_value_func = np.zeros(ns, dtype=np.float32)
    new_value_func = np.zeros(ns, dtype=np.float32)
    iter_num = 0
    for _ in range(max_iterations):
        new_value_func = np.zeros(ns, dtype=np.float32)
        iter_num += 1
        delta = 0
        for s in range(ns):
            a = 0
            temp = 0
            for (prob, nextstate, reward, is_terminal) in env.unwrapped.P[s][a]:
                temp += prob * (reward + gamma * old_value_func[nextstate])
            new_value_func[s] = temp
            for a in range(na):
                temp = 0
                for (prob, nextstate, reward, is_terminal) in env.unwrapped.P[s][a]:
                    temp += prob * (reward + gamma * old_value_func[nextstate])
                new_value_func[s] = max(temp, new_value_func[s])
            delta = max(delta, abs(new_value_func[s]-old_value_func[s]))
        if delta < tol:
            break
        old_value_func = new_value_func
    return new_value_func, iter_num


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)


