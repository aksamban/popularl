import gymnasium as gym
import random
import numpy as np
from gymnasium import Wrapper
from gymnasium.spaces import Box
import torch


def get_cartpole(gravity=True, poleMass=True, length=True, cartMass=True):
    env = gym.make("CartPole-v1")
    # env.seed(42)
    print(type(env.env.env.env))
    env_real = env.env.env.env
    if gravity:
        env_real.gravity = random.uniform(8.0, 12.0)
    if poleMass:
        env_real.masspole = random.uniform(0.05, 0.25)
    if length:
        env_real.length = random.uniform(0.25, 2.0)
    if cartMass:
        env_real.masscart = random.uniform(0.5, 2.0)
    return env


class CustomWrapper(Wrapper):
    def __init__(self, env, embedder):
        super().__init__(env)
        box = Box(shape=(env.observation_space.shape[0]+embedder.embed_dim(),), low=-1000.0, high=1000.0)
        self.observation_space = box
        self.embedder = embedder
        env_real = env.env.env.env
        l = np.concatenate((env.observation_space.sample(),np.array([env.action_space.sample()])))
        self.trajectory = np.array([l])
        self.embedding = np.zeros(embedder.embed_dim(), )
        self.trajectories = {"oar":[], "oa":[]}
        self.all_traject = []
        self.env_features = [env_real.gravity, env_real.masspole, env_real.length, env_real.masscart]
        print(1, self.env_features)

    def step(self, action):
        step_obs, reward, terminated, truncated, info = self.env.step(action)
        # concat step_obs and embedding
        observation = torch.cat((torch.tensor(step_obs), self.embedding))
        self.trajectory.append(np.concatenate((step_obs, np.array([action]))))
        self.all_traject.append(np.concatenate((step_obs, np.array([action, reward]))))
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        original_obs, info = self.env.reset(**kwargs)
        env_real = self.env.env.env.env
        env_real.gravity, env_real.masspole, env_real.length, env_real.masscart = self.env_features
        self.embedding = self.embedder(torch.tensor(np.array(self.trajectory), dtype=torch.float32).unsqueeze(0)).squeeze(0).detach()
        self.trajectories["oar"].append(self.all_traject)
        self.trajectories["oa"].append(self.trajectory)
        self.trajectory = []
        self.all_traject = []
        return torch.cat((torch.tensor(original_obs), self.embedding)), info