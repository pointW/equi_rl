from agents.sac import SAC
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from utils.torch_utils import augmentTransition, perturb

class SACAux(SAC):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False,
                 obs_type='pixel'):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau, alpha, policy_type, target_update_interval,
                         automatic_entropy_tuning, obs_type)

    def calcActorLoss(self):
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()

        pi, log_pi, mean = self.actor.sample(obs)
        self.loss_calc_dict['pi'] = pi
        self.loss_calc_dict['mean'] = mean
        self.loss_calc_dict['log_pi'] = log_pi

        qf1_pi, qf2_pi = self.critic(obs, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        aug_obs = []
        aug_action = mean.detach().cpu().clone().numpy()
        for i in range(batch_size):
            o, _, dxy, _ = perturb(obs[i].detach().cpu().numpy(), None, mean[i][1:3].detach().cpu().numpy(), set_trans_zero=True)
            aug_action[i][1:3] = dxy
            aug_obs.append(o)
        aug_obs = np.stack(aug_obs)
        aug_obs = torch.tensor(aug_obs).to(self.device)
        aug_action = torch.tensor(aug_action).to(self.device)

        _, _, aug_out = self.actor.sample(aug_obs)

        aux_loss = F.mse_loss(aug_out, aug_action)

        policy_loss = policy_loss + aux_loss

        return policy_loss

    def calcCriticLoss(self):
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_obs)
            next_state_log_pi = next_state_log_pi.reshape(batch_size)
            qf1_next_target, qf2_next_target = self.critic_target(next_obs, next_state_action)
            qf1_next_target = qf1_next_target.reshape(batch_size)
            qf2_next_target = qf2_next_target.reshape(batch_size)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + non_final_masks * self.gamma * min_qf_next_target
        qf1, qf2 = self.critic(obs, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1 = qf1.reshape(batch_size)
        qf2 = qf2.reshape(batch_size)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        with torch.no_grad():
            td_error = 0.5 * (torch.abs(qf2 - next_q_value) + torch.abs(qf1 - next_q_value))

        aug_obs = []
        aug_action = action.detach().cpu().clone().numpy()
        for i in range(batch_size):
            o, _, dxy, _ = perturb(obs[i].detach().cpu().numpy(), None, action[i][1:3].detach().cpu().numpy(),
                                   set_trans_zero=True)
            aug_action[i][1:3] = dxy
            aug_obs.append(o)
        aug_obs = np.stack(aug_obs)
        aug_obs = torch.tensor(aug_obs).to(self.device)
        aug_action = torch.tensor(aug_action).to(self.device)

        aug_qf1, aug_qf2 = self.critic(aug_obs, aug_action)
        aug_qf1 = aug_qf1.reshape(batch_size)
        aug_qf2 = aug_qf2.reshape(batch_size)

        qf1_loss = qf1_loss + F.mse_loss(aug_qf1, qf1)
        qf2_loss = qf2_loss + F.mse_loss(aug_qf2, qf2)

        return qf1_loss, qf2_loss, td_error
