import numpy as np
from copy import deepcopy
from torch.optim import Adam
import torch
import core_adv as core
import ggdo2 as ggdo2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:   # 采样后输出为tensor
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k,v in batch.items()}

import torch.nn as nn
import torch.nn.functional as F
class Actor(nn.Module):
    
    def __init__(self, in_dim:int, out_dim:int, init_w:float = 3e-3):
        
        "Initalize"
        
        super(Actor, self).__init__()
        
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128,128)
        self.out = nn.Linear(128,out_dim)
        
       # self.out.weight.data.uniform(-init_w, init_w)
       # self.out.bias.data.uniform(-init_w, init_w)
        
    def forward(self, state:torch.Tensor) -> torch.Tensor:
        
        "Forward method implementation"
        
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        action = self.out(x).tanh()
        
        return action
    
    
class Critic(nn.Module):
    
   def __init__(self,in_dim: int,init_w: float = 3e-3): 
       
        
        "Initialize"
        super(Critic, self).__init__()
        
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128,1)
        
      #  self.out.weight.data.uniform_(-init_w, init_w)
      #  self.out.bias.data.uniform_(-init_w, init_w)

   def forward(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
      "Foward method implementation"
      x = torch.cat((state, action), dim=-1)
      x = F.relu(self.hidden1(x))
      x = F.relu(self.hidden2(x))
      
      value = self.out(x)
      
      return value
  

import torch.optim as optim

class SGLD_DDPG:
    def __init__(self, alpha, obs_dim, action_dim, act_bound,  
                replay_size=int(1e6), gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, act_noise=0.1):

        self.actor = Actor(obs_dim, action_dim).to(device)
        self.actor_target = Actor(obs_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.gamma = gamma
        self.alpha = alpha
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor_bar = Actor(obs_dim, action_dim).to(device)
        self.actor_outer = Actor(obs_dim, action_dim).to(device)

        self.critic = Critic(obs_dim + action_dim).to(device)
        self.critic_target = Critic(obs_dim + action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.action_dim = action_dim
        

        self.actor_optim = ggdo2.GGDO(self.actor.parameters(), 0.01, momentum=0.9,
                        weight_decay=5e-4)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)


        self.adversary = Actor(obs_dim, action_dim).to(device)
        self.adversary_target = Actor(obs_dim, action_dim).to(device)
        self.adversary_bar = Actor(obs_dim, action_dim).to(device)
        self.adversary_outer = Actor(obs_dim, action_dim).to(device)


        self.adversary_optim = ggdo2.GGDO(self.adversary.parameters(), 0.01, noise=0.1,weight_decay = 5e-4)


        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=action_dim, size=replay_size)

  
    def update(self, samples):
        
        state_batch = torch.as_tensor(samples["obs"], device = self.device)
        next_state_batch = torch.as_tensor(samples["obs2"], device = self.device)
        action_batch = torch.as_tensor(samples["act"], device = self.device)
        reward_batch = torch.as_tensor(samples["rew"].reshape(-1, 1), device = self.device)
        done_batch = torch.as_tensor(samples["done"].reshape(-1, 1), device = self.device)
        
        masks_batch = 1- done_batch
    
        "Train critic"
        values = self.critic(state_batch, action_batch)
        
        value_loss = 0
        policy_loss = 0
        adversary_loss = 0
        
        next_action_batch = (1 - self.alpha) *self.actor_target(next_state_batch) + self.alpha * self.adversary_target(next_state_batch)
        
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)
            
        expected_state_action_batch = reward_batch +self. gamma * masks_batch * next_state_action_values

        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)

        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optimizer.step()
        _value_loss = value_loss.item()
        
        "Train adversary"
        self.adversary_optim.zero_grad() 
        with torch.no_grad():
                real_action = self.actor_outer(next_state_batch)
            
        action = (1 - self.alpha) * real_action + self.alpha * self.adversary(next_state_batch)
        adversary_loss = self.critic(state_batch, action)
        adversary_loss = adversary_loss.mean()
        adversary_loss.backward()
        self.adversary_optim.step()
        _adversary_loss = adversary_loss.item()
        
        "Train actor"
        
        self.actor_optim.zero_grad()
        with torch.no_grad():
                self.adversary_action = self.adversary_outer(next_state_batch)
        action = (1 - self.alpha) * self.actor(next_state_batch) + self.alpha * self.adversary_action
        policy_loss = -self.critic(state_batch, action)
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()
        _policy_loss = policy_loss.item()
        
        
        "Accumulation"
        value_loss += _value_loss
        policy_loss += _policy_loss
        adversary_loss += _adversary_loss
        
      #  critic_losses_arr.append(float(crit_loss_arr))     
      #  act_loss_arr = actor_loss.detach().cpu().numpy()
        
      #  actor_losses_arr.append(float(act_loss_arr))
        "Inner update"  
      
        beta_up = 0.9      
        
        for target_param, param in zip(self.actor_bar.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - beta_up) + param.data * beta_up)
        
        for target_param, param in zip(self.adversary_bar.parameters(), self.adversary.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - beta_up) + param.data * beta_up)
        
        "Soft Update"
        
        tau = 0.01
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
        for target_param, param in zip(self.adversary_target.parameters(), self.adversary.parameters()):
             target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
             target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        
        "Outer update"
        
       
        for target_param, param in zip(self.actor_outer.parameters(), self.actor_bar.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - beta_up) + param.data * beta_up)
        
        for target_param, param in zip(self.adversary_outer.parameters(), self.adversary_bar.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - beta_up) + param.data * beta_up)
        
    
    

    def get_action(self,state: np.ndarray, noise_scale):
    
        "Select an action from the input state"
        # if initial random action to be conducted
    

        selected_action = self.actor_outer(torch.FloatTensor(state).to(device)).detach().cpu().numpy()
        
        selected_action += noise_scale * np.random.randn(self.action_dim)
        selected_action_mu = np.clip(selected_action, -1.0, 1.0)
        
        selected_action_mu = selected_action_mu*(1-self.alpha)
        
        
        
        adv_mu = self.adversary(torch.FloatTensor(state).to(device)).detach().cpu().numpy()
        adv_mu += noise_scale * np.random.randn(self.action_dim)
        adv_mu = np.clip(adv_mu, -1.0, 1.0)
        
        adv_mu = adv_mu*(self.alpha)        
        

        selected_action_mu += adv_mu
    
        return selected_action_mu
    
    def load_from(self, agent):
        
        self.ac.pi.load_state_dict(agent.ac.pi.state_dict())
        self.ac_targ.pi.load_state_dict(agent.ac_targ.pi.state_dict())
        
        self.ac.q.load_state_dict(agent.ac.q.state_dict())
        self.ac_targ.q.load_state_dict(agent.ac_targ.q.state_dict())
        
        self.ac.adv.load_state_dict(agent.ac.adv.state_dict())
        self.ac_targ.adv.load_state_dict(agent.ac_targ.adv.state_dict())
        
       

        
        
        
        
        
    
