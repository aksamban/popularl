from envs.randpole_gen import get_cartpole, CustomWrapper
from models.task_embedder import TaskEmbedder
from models.grad_callback import CustomCallback
from models.vae import VAE
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# Function to monitor policy weights
def print_policy_weights(policy, prefix="Initial"):
    print(f"{prefix} Policy Weights:")
    for name, param in policy.named_parameters():
        print(f"{name}: {param.data.norm()}")  # Print the norm of each parameter


def train(save_dir, log_dir, beta, num_meta_iterations=10, warmup_iterations=6, seed=42,
          num_tasks_per_meta_iteration=5, timesteps_per_task=3000, vae_rollout_size=500, latent_size= 8,
          embedder_hidden_size=32, embedding_size=8, observation_size=4, action_size=1, vae_hidden_size=8):

    # embedder of input 5, hidden 32, output 8
    embedder = TaskEmbedder(action_size+observation_size, embedder_hidden_size, embedding_size)  # o_t: (4,), a_t: (1,)
    embed_optimizer = torch.optim.Adam(embedder.parameters(), lr=1e-3)
    # embedder of input 5 (ot at), hidden 32, latent 8, grounding 8, output 6 (ot at rt)
    vae = VAE(action_size+observation_size, vae_hidden_size, latent_size, embedding_size, vae_hidden_size, observation_size+action_size+1)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    # Initialize PPO policy
    #t = int(time.time())
    #log_dir = f'logs/ppo_tensorboard_logs_{t}/'
    #save_dir = f'model_logs/{t}/'
    initial_env = DummyVecEnv([lambda: CustomWrapper(get_cartpole(), embedder, torch.zeros(latent_size))])
    policy = PPO("MlpPolicy", initial_env, verbose=4, seed=seed, tensorboard_log=log_dir)
    meta_optimizer = torch.optim.Adam(policy.policy.parameters(), lr=1e-3)

    for meta_iter in range(num_meta_iterations):
        print(f"Meta Iteration {meta_iter + 1}/{num_meta_iterations}")

        # Collect task gradients for meta-update
        task_gradients = []
        # Loop over tasks within the current meta-iteration
        for task in range(num_tasks_per_meta_iteration):
            print(f"  Task {task + 1}/{num_tasks_per_meta_iteration}")
            
            # Dummy Z
            z = torch.zeros(latent_size)
            en = get_cartpole()
            env = DummyVecEnv([lambda: CustomWrapper(en, embedder, z)])
            policy.set_env(env)
            print(policy.env.envs[0].env.env.env.env.gravity)
            if meta_iter >= warmup_iterations:
                # Get Latent Vectors Z
                print("USING LATENT VARIABLE")
                #policy._last_obs = np.array([np.zeros(20)])
                #policy._last_obs =  policy.env.reset()

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
        
                env = DummyVecEnv([lambda: CustomWrapper(en, embedder, z)])
                policy.set_env(env)

            # Perform task-specific learning using manual updates
            callback = CustomCallback(embedder, embed_optimizer, z, observation_size)
            policy.learn(total_timesteps=timesteps_per_task, reset_num_timesteps=False, callback=callback)

            #VAE TRAINING
            if meta_iter+1 >= warmup_iterations:
                #print("DOING VAE TRAINING")
                trajectories_oar = policy.env.envs[0].trajectories["oar"]
                trajectories_oa = policy.env.envs[0].trajectories["oa"]
                embedding = policy.env.envs[0].embedding.unsqueeze(0)
                for i in range(len(trajectories_oa)):
                    if i!= 0:
                        oa = trajectories_oa[i]
                        recon, mu, var = vae.forward(torch.tensor(oa, dtype=torch.float32), embedding, len(oa))
                        oar = trajectories_oar[i]
                        vae_optimizer.zero_grad()
                        loss = vae.loss(recon, torch.tensor(oar, dtype=torch.float32).unsqueeze(0), mu, var, beta)
                        loss.backward()
                        vae_optimizer.step()

            # Collect task gradients for meta-update
            task_gradients.append([param.grad.clone() for param in policy.policy.parameters()])  # Clone to keep track of gradients

        # Step 3: Outer loop - Meta-update the policy using gradients from all tasks
        meta_optimizer.zero_grad()

        # Example: Average the task gradients (you can use other meta-optimization schemes)
        for task_gradient in task_gradients:
            for param, grad in zip(policy.policy.parameters(), task_gradient):
                if param.grad is None:
                    param.grad = grad  # Initialize the gradient for the first time
                else:
                    param.grad.add_(grad)  # Accumulate gradients from all tasks

        # Perform the meta-update step
        meta_optimizer.step()

        # Optionally, print out the current status of the policy's training
        print(f"Meta Update Completed for Iteration {meta_iter + 1}")
        os.makedirs(save_dir+f'meta{meta_iter+1}/')
        torch.save({
                'PPO': policy.policy.state_dict(),
                'Meta_Optimizer': meta_optimizer.state_dict(),
                'Embedder': embedder.state_dict(),
                'Embed_Optimizer': embed_optimizer.state_dict(),
                'VAE': vae.state_dict(),
                'VAE_Optimizer': vae_optimizer.state_dict(),
                }, save_dir+f'meta{meta_iter+1}.pth')

        # Optionally, inspect the meta-updated policy weights
        print_policy_weights(policy.policy, prefix=f"Meta-Updated after Iteration {meta_iter + 1}")

