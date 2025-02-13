from envs.randpole_gen_maml import CustomWrapper
from models.task_embedder import TaskEmbedder
from models.grad_callback_maml import CustomCallback
import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def test_maml(load_dir, env_values, save_dir, log_dir, seed=42, timesteps_per_task=1000, vae_rollout_size=500,
              embedder_hidden_size=32, embedding_size=8, observation_size=4, action_size=1):
    
    """
    test_function does blah blah blah.

    :param load_dir: directory to load weights from
    :param env_values: list of length 4 [gravity (8:12), pole mass (0.05:0.25), pole length (0.25:2), cart mass (0.5:2)]
    """ 

    # embedder of input 5, hidden 32, output 8
    embedder = TaskEmbedder(action_size+observation_size, embedder_hidden_size, embedding_size)  # o_t: (4,), a_t: (1,)
    embed_optimizer = torch.optim.Adam(embedder.parameters(), lr=1e-3)
    checkpoint = torch.load(load_dir, weights_only=True)

    env = gym.make('CartPole-v1')
    env_real = env.env.env.env
    env_real.gravity = env_values[0]  # trained on 8:12
    env_real.masspole = env_values[1]  # trained on 0.05:0.25
    env_real.length = env_values[2]  # trained on 0.25:2
    env_real.masscart = env_values[3] # trained on 0.5:2

    initial_env = DummyVecEnv([lambda: CustomWrapper(env, embedder)])
    policy = PPO("MlpPolicy", initial_env, verbose=4, seed=seed, tensorboard_log=log_dir)
    meta_optimizer = torch.optim.Adam(policy.policy.parameters(), lr=1e-3)

    policy.policy.load_state_dict(checkpoint['PPO'])
    meta_optimizer.load_state_dict(checkpoint['Meta_Optimizer'])
    embedder.load_state_dict(checkpoint['Embedder'])
    embed_optimizer.load_state_dict(checkpoint['Embed_Optimizer'])

    total_timesteps, callback = policy._setup_learn(
        total_timesteps=vae_rollout_size,
        callback=None,
        reset_num_timesteps=False,
        tb_log_name='',
        progress_bar=False,
    )

    continue_training = policy.collect_rollouts(env=policy.env, callback=callback, rollout_buffer=policy.rollout_buffer, n_rollout_steps=vae_rollout_size)
    trajectories_oa = policy.env.envs[0].trajectories["oa"]
    embedding = policy.env.envs[0].embedding.unsqueeze(0)
    oa = trajectories_oa[-2]

    callback = CustomCallback(embedder, embed_optimizer)
    policy.learn(total_timesteps=timesteps_per_task, reset_num_timesteps=False, callback=callback)


    trajectories_oar = policy.env.envs[0].trajectories["oar"]
    print("No. Of Episodes: ", len(trajectories_oar))
    sum = 0
    for i in range(len(trajectories_oar)):
        if i!= 0:
            oar = trajectories_oar[i]
            sum += np.array(oar, dtype=np.float32).shape[0]
    sum /= len(trajectories_oar)-1
    print("Average Ep Len: ", sum)

    torch.save({
            'PPO': policy.policy.state_dict(),
            'Meta_Optimizer': meta_optimizer.state_dict(),
            'Embedder': embedder.state_dict(),
            'Embed_Optimizer': embed_optimizer.state_dict(),
            }, save_dir)