import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class RSSMCell(nn.Module):
    def __init__(self, deter_dim, stoch_dim, hidden_dim, action_dim):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.action_dim = action_dim
        # Input to GRU: previous stochastic state + previous deterministic state + action
        self.gru = nn.GRUCell(stoch_dim + deter_dim + action_dim, deter_dim)
        # Prior network: p(z_t | h_{t-1})
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)
        )
        # Posterior network: q(z_t | h_{t-1}, x_t)
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),   # we'll pass encoded obs separately, but x_t is encoded
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)
        )

    def forward(self, prev_deter, prev_stoch, action, encoded_obs):
        # prev_deter: (batch, deter_dim)
        # prev_stoch: (batch, stoch_dim)
        # action: (batch, action_dim)
        # encoded_obs: (batch, stoch_dim)   # encoded observation to the same dim as stoch
        gru_input = torch.cat([prev_stoch, prev_deter, action], dim=-1)
        deter = self.gru(gru_input, prev_deter)
        # Prior
        prior_params = self.prior_net(deter)
        prior_mean, prior_logstd = prior_params.chunk(2, dim=-1)
        prior_std = torch.exp(prior_logstd)
        # Posterior using encoded observation
        post_input = torch.cat([deter, encoded_obs], dim=-1)
        post_params = self.post_net(post_input)
        post_mean, post_logstd = post_params.chunk(2, dim=-1)
        post_std = torch.exp(post_logstd)
        # Sample stochastic state (reparameterized)
        eps = torch.randn_like(post_mean)
        stoch = post_mean + eps * post_std
        # KL divergence
        kl = (Normal(post_mean, post_std).log_prob(stoch) -
              Normal(prior_mean, prior_std).log_prob(stoch)).sum(dim=-1).mean()
        return deter, stoch, prior_mean, prior_std, post_mean, post_std, kl

class RSSM(nn.Module):
    def __init__(self, deter_dim, stoch_dim, hidden_dim, action_dim):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.rssm_cell = RSSMCell(deter_dim, stoch_dim, hidden_dim, action_dim)
        self.register_buffer('init_deter', torch.zeros(1, deter_dim))
        self.register_buffer('init_stoch', torch.zeros(1, stoch_dim))

    def initial_state(self, batch_size):
        deter = self.init_deter.expand(batch_size, -1).contiguous()
        stoch = self.init_stoch.expand(batch_size, -1).contiguous()
        return deter, stoch

    def forward(self, encoded_obs_seq, action_seq, batch_size, seq_len):
        deter, stoch = self.initial_state(batch_size)
        kl_sum = 0.0
        # Store states for later
        deters = [deter]
        stochs = [stoch]
        for t in range(seq_len):
            deter, stoch, _, _, _, _, kl = self.rssm_cell(deter, stoch, action_seq[:, t], encoded_obs_seq[:, t])
            kl_sum += kl
            deters.append(deter)
            stochs.append(stoch)
        return kl_sum / seq_len, torch.stack(deters[1:], dim=1), torch.stack(stochs[1:], dim=1)

    def imagine(self, initial_deter, initial_stoch, actions, seq_len):
        """Rollout policy actions in latent space."""
        deter = initial_deter
        stoch = initial_stoch
        states = []
        for t in range(seq_len):
            # Prior only (no observation)
            gru_input = torch.cat([stoch, deter, actions[:, t]], dim=-1)
            deter = self.rssm_cell.gru(gru_input, deter)
            prior_params = self.rssm_cell.prior_net(deter)
            prior_mean, prior_logstd = prior_params.chunk(2, dim=-1)
            prior_std = torch.exp(prior_logstd)
            eps = torch.randn_like(prior_mean)
            stoch = prior_mean + eps * prior_std
            states.append((deter, stoch))
        return states

class Encoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, obs):
        return self.net(obs)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, deter, stoch):
        latent = torch.cat([deter, stoch], dim=-1)
        return self.net(latent)

class RewardPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, deter, stoch):
        latent = torch.cat([deter, stoch], dim=-1)
        return self.net(latent)

class WorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, deter_dim, stoch_dim, hidden_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.encoder = Encoder(obs_dim, hidden_dim, stoch_dim)
        self.decoder = Decoder(deter_dim + stoch_dim, hidden_dim, obs_dim)
        self.rssm = RSSM(deter_dim, stoch_dim, hidden_dim, action_dim)
        self.reward_pred = RewardPredictor(deter_dim + stoch_dim, hidden_dim)

    def forward(self, obs_seq, action_seq):
        batch, seq_len, _ = obs_seq.shape
        # Encode observations
        encoded = self.encoder(obs_seq.view(-1, self.obs_dim)).view(batch, seq_len, -1)
        # Run RSSM
        kl, deters, stochs = self.rssm(encoded, action_seq, batch, seq_len)
        # Decode observations and predict rewards for each step (for training)
        # (We'll compute loss inside trainer)
        return kl, deters, stochs

    def compute_loss(self, obs_seq, action_seq):
        kl, deters, stochs = self.forward(obs_seq, action_seq)
        # Reconstruct observations
        recon = self.decoder(deters.reshape(-1, deters.shape[-1]), stochs.reshape(-1, stochs.shape[-1])).reshape(obs_seq.shape)
        recon_loss = nn.MSELoss()(recon, obs_seq)
        # Predict rewards (we don't have rewards in data; use next-day return? We'll use daily return as reward)
        # For simplicity, we'll compute reward prediction loss using actual daily returns (from observation).
        # We'll treat the last element of observation as reward? Actually we need separate reward signal.
        # In this simplified version, we'll skip reward prediction for world model training,
        # and only use KL + reconstruction.
        # For full Dreamer, we would also predict reward. We'll add that in trainer using realized returns.
        return recon_loss + 0.1 * kl

class Actor(nn.Module):
    def __init__(self, latent_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, deter, stoch):
        latent = torch.cat([deter, stoch], dim=-1)
        return self.net(latent)

class Critic(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, deter, stoch):
        latent = torch.cat([deter, stoch], dim=-1)
        return self.net(latent)
