import gymnasium as gym
import sys

# env = gym.make('PointMaze_UMazeDense-v3')
"""Random policy on an environment."""

import numpy as np
import argparse


def get_goal_sample_fn(env_name, evaluate):
    if env_name == "PointMaze":
        if evaluate:
            return lambda: np.array([-1.0, -1.0])
        else:
            return lambda: np.array([-1.0, -1.0])
        assert False, "Unknown env"


def get_reward_fn(env_name):
    if env_name == "PointMaze":
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
        assert False, "Unknown env"


def success_fn(last_reward):
    return last_reward > -0.8


class EnvWithGoal(object):
    def __init__(self, base_env, env_name):
        self.base_env = base_env
        self.env_name = env_name
        self.evaluate = False
        self.reward_fn = get_reward_fn(env_name)
        self.goal = None
        self.distance_threshold = 0.8
        self.count = 0
        self.state_dim = 4 + 1
        self.action_dim = 2

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self):
        # self.viewer_setup()
        options = {  # related to U_maze in point_maze
            "goal_cell": np.array([5, 3]),  # target
            "reset_cell": np.array([1, 1]),  # init
        }
        if self.evaluate:
            obs, _ = self.base_env.reset(options=options)
        else:
            obs, _ = self.base_env.reset()

        self.count = 0
        self.goal = obs["desired_goal"]
        obs["observation"] = np.r_[obs["observation"].copy(), self.count]
        return obs
        # return {
        #     # add timestep
        #     'observation': np.r_[obs.copy(), self.count],
        #     'achieved_goal': obs,
        #     'desired_goal': self.goal,
        # }

    def step(self, a):
        obs, _, done, info, __ = self.base_env.step(a)
        self.count += 1
        obs["observation"] = np.r_[obs["observation"].copy(), self.count]
        reward = self.reward_fn(obs["observation"], self.goal)
        # next_obs =   {
        #     'observation': np.r_[obs['observation'].copy(), self.count],
        #     # add timestep
        #      'achieved_goal': obs,
        #      'desired_goal': self.goal}

        return obs, reward, done or self.count >= 350, info

    def render(self):
        self.base_env.render()

    def get_image(self):
        self.render()
        data = self.base_env.viewer.get_image()

        img_data = data[0]
        width = data[1]
        height = data[2]

        tmp = np.fromstring(img_data, dtype=np.uint8)
        image_obs = np.reshape(tmp, [height, width, 3])
        image_obs = np.flipud(image_obs)

        return image_obs

    def close(self):
        self.base_env.close()

    @property
    def action_space(self):
        return self.base_env.action_space

    @property
    def observation_space(self):
        return self.base_env.observation_space


def run_environment(env_name, episode_length, num_episodes):
    env = EnvWithGoal(gym.make("PointMaze_UMazeDense-v3"), env_name)
    next_obs = env.reset()
    # print(env.base_env)
    scale = env.action_space.high * np.ones(env.action_dim)
    print(scale)
    print(next_obs["observation"])
    for i in range(500):
        # env.render()
        a = np.array((0.5, 0.5), dtype=float)
        next_obs, reward, done, info = env.step(a)

        #
        #     print(a)
        #  print(next_obs['observation'])
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="PointMaze", type=str)
    parser.add_argument("--episode_length", default=500, type=int)
    parser.add_argument("--num_episodes", default=100, type=int)

    args = parser.parse_args()
    print(args.env_name)
    run_environment(args.env_name, args.episode_length, args.num_episodes)
