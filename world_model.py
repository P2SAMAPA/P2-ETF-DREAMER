import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RSSMCell(nn.Module):
    """Recurrent State Space Model cell."""
    def __init__(self, deter_dim, stoch_dim, hidden_dim):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim
        # GRU for deterministic state
        self.gru = nn.GRUCell(stoch_dim + deter_dim, deter_dim)
        # Prior distribution: p(z_t | h_{t-1})
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)
        )
        # Posterior distribution: q(z_t | h_{t-1}, x_t)
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + deter_dim, hidden_dim),  # concatenate h_prev and h? Actually x_t encoded
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)
        )
    def forward(self, prev_deter, prev_stoch, encoded_obs):
        # prev_deter: (batch, deter_dim)
        # prev_stoch: (batch, stoch_dim)
        h_prev = prev_deter
        # Compute deterministic state
        gru_input = torch.cat([prev_stoch, h_prev], dim=-1)
        deter = self.gru(gru_input, h_prev)
        # Prior distribution
        prior_params = self.prior_net(deter)
        prior_mean, prior_logstd = prior_params.chunk(2, dim=-1)
        prior_std = torch.exp(prior_logstd)
        # Posterior uses encoded observation
        post_input = torch.cat([deter, encoded_obs], dim=-1)
        post_params = self.post_net(post_input)
        post_mean, post_logstd = post_params.chunk(2, dim=-1)
        post_std = torch.exp(post_logstd)
        # Sample stochastic state (reparameterized)
        eps = torch.randn_like(post_mean)
        stoch = post_mean + eps * post_std
        # KL divergence between posterior and prior
        kl = (torch.distributions.Normal(post_mean, post_std).log_prob(stoch) -
              torch.distributions.Normal(prior_mean, prior_std).log_prob(stoch)).sum(dim=-1).mean()
        return deter, stoch, prior_mean, prior_std, post_mean, post_std, kl

class RSSM(nn.Module):
    def __init__(self, deter_dim, stoch_dim, hidden_dim):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.rssm_cell = RSSMCell(deter_dim, stoch_dim, hidden_dim)
        self.register_buffer('init_deter', torch.zeros(1, deter_dim))
        self.register_buffer('init_stoch', torch.zeros(1, stoch_dim))
    def initial_state(self, batch_size):
        deter = self.init_deter.expand(batch_size, -1).contiguous()
        stoch = self.init_stoch.expand(batch_size, -1).contiguous()
        return deter, stoch
    def forward(self, encoded_obs_seq, batch_size, seq_len):
        # encoded_obs_seq: (batch, seq_len, enc_dim)
        deter, stoch = self.initial_state(batch_size)
        kl_sum = 0.0
        for t in range(seq_len):
            deter, stoch, _, _, _, _, kl = self.rssm_cell(deter, stoch, encoded_obs_seq[:, t])
            kl_sum += kl
        return kl_sum / seq_len

class Encoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
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

class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, deter_dim, stoch_dim, hidden_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.encoder = Encoder(obs_dim, hidden_dim, stoch_dim)  # encode to stochastic dim
        self.decoder = Decoder(deter_dim+stoch_dim, hidden_dim, obs_dim)
        self.rssm = RSSM(deter_dim, stoch_dim, hidden_dim)
        self.reward_pred = nn.Sequential(
            nn.Linear(deter_dim+stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, obs_seq, action_seq):
        # obs_seq: (batch, seq_len, obs_dim)
        batch, seq_len, _ = obs_seq.shape
        encoded = self.encoder(obs_seq.reshape(-1, self.obs_dim)).reshape(batch, seq_len, -1)
        # RSSM dynamics (only for training KL loss)
        kl = self.rssm(encoded, batch, seq_len)
        # Decode to reconstruct observations (for each step)
        # We'll also compute reward predictions
        # This is simplified; full world model also uses actions as input to dynamics. We'll extend.
        return kl

class Actor(nn.Module):
    def __init__(self, latent_dim, hidden_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1)   # portfolio weights long only
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
