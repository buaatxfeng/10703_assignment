import deeprl_hw1.rl as rl
import deeprl_hw1.lake_envs as lake_env
import gym
import time

env1 = gym.make('Stochastic-8x8-FrozenLake-v0')
action_names = lake_env.action_names
gamma = 0.9
policy, value_func, policy_iter_num, value_iter_num = rl.policy_iteration(env1, gamma, max_iterations=int(1e3))
new_value, num = rl.value_iteration(env1, gamma, max_iterations=int(1e3), tol=1e-3)
print(policy_iter_num)
print(value_iter_num)
rl.print_policy(policy, action_names)