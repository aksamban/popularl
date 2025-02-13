from envs.randpole_gen import get_cartpole, CustomWrapper
from models.task_embedder import TaskEmbedder
from models.grad_callback import CustomCallback
from models.vae import VAE
import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def test(load_dir, env_values, save_dir, log_dir, seed=42, timesteps_per_task=3000, vae_rollout_size=500, latent_size= 8,
         embedder_hidden_size=32, embedding_size=8, observation_size=4, action_size=1, vae_hidden_size=8):

    # embedder of input 5, hidden 32, output 8
    embedder = TaskEmbedder(action_size+observation_size, embedder_hidden_size, embedding_size)  # o_t: (4,), a_t: (1,)
    embed_optimizer = torch.optim.Adam(embedder.parameters(), lr=1e-3)
    # embedder of input 5 (ot at), hidden 32, latent 8, grounding 8, output 6 (ot at rt)
    vae = VAE(action_size+observation_size, vae_hidden_size, latent_size, embedding_size, vae_hidden_size, observation_size+action_size+1)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    checkpoint = torch.load(load_dir, weights_only=True)

    env = gym.make('CartPole-v1')
    env_real = env.env.env.env
    #env_values = [13, 0.3, 3.0, 0.4]
    env_real.gravity = env_values[0]  # trained on 8:12
    env_real.masspole = env_values[1]  # trained on 0.05:0.25
    env_real.length = env_values[2]  # trained on 0.25:2
    env_real.masscart = env_values[3] # trained on 0.5:2

    #log_dir = f'experiments/200k/length2/vae/tensorboard/{env_values[0]}_{env_values[1]}_{env_values[2]}_{env_values[3]}/'
    #save_dir = f'experiments/200k/length2/vae/policies/{env_values[0]}_{env_values[1]}_{env_values[2]}_{env_values[3]}.pth'

    #log_dir = f'experiments/summa/vae/{env_values[0]}_{env_values[1]}_{env_values[2]}_{env_values[3]}/'
    #save_dir = f'experiments/summa/vae/{env_values[0]}_{env_values[1]}_{env_values[2]}_{env_values[3]}.pth'

    initial_env = DummyVecEnv([lambda: CustomWrapper(env, embedder, torch.zeros(latent_size))])
    policy = PPO("MlpPolicy", initial_env, verbose=4, seed=seed, tensorboard_log=log_dir)
    meta_optimizer = torch.optim.Adam(policy.policy.parameters(), lr=1e-3)

    policy.policy.load_state_dict(checkpoint['PPO'])
    meta_optimizer.load_state_dict(checkpoint['Meta_Optimizer'])
    embedder.load_state_dict(checkpoint['Embedder'])
    embed_optimizer.load_state_dict(checkpoint['Embed_Optimizer'])
    vae.load_state_dict(checkpoint['VAE'])
    vae_optimizer.load_state_dict(checkpoint['VAE_Optimizer'])

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
    z = vae.latent(torch.tensor(oa, dtype=torch.float32))

    callback = CustomCallback(embedder, embed_optimizer, z)
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
            'VAE': vae.state_dict(),
            'VAE_Optimizer': vae_optimizer.state_dict(),
            }, save_dir)