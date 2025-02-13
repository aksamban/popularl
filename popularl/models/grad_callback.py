import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn.functional as F


class DummyCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        pass


class CustomCallback(BaseCallback):
    def __init__(self, embedding_module, embedding_optimizer, z, observation_size, verbose=0):
        super().__init__(verbose)
        self.embedding_module = embedding_module
        self.optimizer = embedding_optimizer
        self.z = z
        self.observation_size = observation_size

    def _on_step(self) -> bool:
        """ This method runs at every step during training. """
        return True
    
    def _on_rollout_end(self):
        """
        This method runs after a rollout ends. We use PPO's loss calculation to backpropagate
        through the embedding module.
        """
        # Access the latest batch of observations from the model's rollout buffer
        buffer = self.model.rollout_buffer
        
        # Get the batch of observations and actions
        # actions, observvations, advantages, old_log_prob, old_values, returns
        batch_obs = buffer.observations  # Shape: (batch_size, n_envs, observation_size)
        batch_act = buffer.actions  # Shape: (batch_size, n_envs, action_size)
        batch_adv = buffer.advantages
        batch_ret = buffer.returns
        batch_rew = buffer.rewards
        batch_old_log_prob = buffer.log_probs
        batch_old_values = buffer.values
        episode_starts = buffer.episode_starts  # Shape: (batch_size, n_envs)

        current_episode_obs_act = []
        current_episode_obs = []
        current_episode_act = []
        current_episode_adv = []
        current_episode_ret = [batch_ret[0][0]]
        # np.sum(sum(n[0] for n in batch_rew))
        old_log_prob = []
        old_values = []
        embedding = torch.zeros(self.embedding_module.embed_dim()).unsqueeze(0)
        
        losses = []
        episode_n = 0

        for i in range(len(episode_starts)):

            if episode_starts[i][0] == 1 and i != 0 and episode_n != 0:
                embed_input = torch.tensor(np.array(current_episode_obs_act, dtype=np.float32)).unsqueeze(0)
                # get embedding
                next_embedding = self.embedding_module(embed_input)


                # step wise
                observations = torch.tensor(current_episode_obs, dtype=torch.float32)
                embeddings = embedding.squeeze(0).repeat(len(current_episode_obs))
                latents = self.z.repeat(len(current_episode_obs))
                concat_input = torch.cat((observations, embeddings, latents))
                actions = torch.tensor(current_episode_act)
                values, log_prob, entropy = self.model.policy.evaluate_actions(concat_input, actions)
                values = values.flatten()
                advantages = torch.tensor(current_episode_adv)
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - torch.tensor(old_log_prob))
                policy_loss = (advantages * ratio).mean()
                value_loss = F.mse_loss(torch.tensor(current_episode_ret), values)
                loss = policy_loss + self.model.vf_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                current_episode_obs_act = []
                current_episode_obs = []
                current_episode_act = []
                current_episode_adv = []
                current_episode_ret = [batch_ret[i][0]]
                old_log_prob = []
                old_values = []
                embedding = next_embedding
                episode_n += 1
            
            current_episode_obs_act.append(np.concatenate((batch_obs[i][0][:self.observation_size], batch_act[i][0])))
            current_episode_obs.append(batch_obs[i][0][:self.observation_size])
            current_episode_act.append(batch_act[i])
            current_episode_adv.append(batch_adv[i])
            # current_episode_ret.append(batch_ret[i])
            old_log_prob.append(batch_old_log_prob[i])
            old_values.append(batch_old_values[i])
        
        # print(losses)
        return True
