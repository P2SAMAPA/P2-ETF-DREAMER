import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from world_model import WorldModel, Actor, Critic

def prepare_sequences(data_array, sequence_length=20):
    T, n = data_array.shape
    sequences = []
    for i in range(0, T - sequence_length, sequence_length):
        seq = data_array[i:i+sequence_length]
        sequences.append(seq)
    return np.array(sequences)

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t] if t+1 < len(values) else rewards[t] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages, dtype=torch.float32)

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (DreamerV3) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < config.TRAIN_WINDOW + 100:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Training data: last TRAIN_WINDOW days
        train_returns = returns.iloc[-config.TRAIN_WINDOW:].values
        obs_mean = train_returns.mean(axis=0)
        obs_std = train_returns.std(axis=0) + 1e-8
        train_obs = (train_returns - obs_mean) / obs_std
        n_assets = train_obs.shape[1]
        action_dim = n_assets

        # Build sequences for world model
        seq_len = config.SEQUENCE_LENGTH
        sequences = prepare_sequences(train_obs, seq_len)
        if len(sequences) < 2:
            print("  Not enough sequences")
            continue
        obs_tensor = torch.tensor(sequences, dtype=torch.float32)
        # Dummy actions (uniform for initial world model training)
        action_seq = torch.full((obs_tensor.shape[0], seq_len, action_dim), 1.0/action_dim)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_model = WorldModel(
            obs_dim=n_assets,
            action_dim=action_dim,
            deter_dim=config.DETER_DIM,
            stoch_dim=config.STOCH_DIM,
            hidden_dim=config.HIDDEN_DIM
        ).to(device)
        wm_optim = optim.Adam(world_model.parameters(), lr=config.WM_LEARNING_RATE)

        print("  Training world model...")
        obs_tensor = obs_tensor.to(device)
        action_seq = action_seq.to(device)
        for epoch in range(config.WM_EPOCHS):
            indices = np.random.permutation(len(obs_tensor))
            total_loss = 0.0
            for i in range(0, len(indices), config.WM_BATCH_SIZE):
                batch_idx = indices[i:i+config.WM_BATCH_SIZE]
                batch_obs = obs_tensor[batch_idx]
                batch_act = action_seq[batch_idx]
                loss = world_model.compute_loss(batch_obs, batch_act)
                wm_optim.zero_grad()
                loss.backward()
                wm_optim.step()
                total_loss += loss.item()
            if (epoch+1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{config.WM_EPOCHS}, loss: {total_loss/len(indices):.4f}")

        # Actor and critic networks
        latent_dim = config.DETER_DIM + config.STOCH_DIM
        actor = Actor(latent_dim, config.ACTOR_HIDDEN, action_dim).to(device)
        critic = Critic(latent_dim, config.ACTOR_HIDDEN).to(device)
        actor_optim = optim.Adam(actor.parameters(), lr=config.ACTOR_LEARNING_RATE)
        critic_optim = optim.Adam(critic.parameters(), lr=config.ACTOR_LEARNING_RATE)

        # Get initial latent state from last real observation
        last_obs = train_obs[-1:]
        last_obs_t = torch.tensor(last_obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            encoded = world_model.encoder(last_obs_t)
            deter = world_model.rssm.init_deter.expand(1, -1).to(device)
            stoch = world_model.rssm.init_stoch.expand(1, -1).to(device)
            zero_action = torch.zeros(1, action_dim).to(device)
            _, stoch, _, _, _, _, _ = world_model.rssm.rssm_cell(deter, stoch, zero_action, encoded)
            initial_deter = deter
            initial_stoch = stoch

        print("  Training actor-critic on imagined rollouts...")
        for epoch in range(config.ACTOR_EPOCHS):
            # Generate one batch of imagined rollouts from the initial latent state
            # We'll generate multiple rollouts using different random seeds (same initial state)
            batch_rollouts = []
            for _ in range(config.ACTOR_BATCH_SIZE):
                deter = initial_deter.clone()
                stoch = initial_stoch.clone()
                log_probs = []
                rewards = []
                values = []
                actions = []
                for t in range(config.ROLLOUT_LENGTH):
                    action_probs = actor(deter, stoch)  # (1, action_dim)
                    # Sample from distribution (categorical softmax)
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()  # scalar index
                    log_prob = dist.log_prob(action)
                    # Convert action to one-hot for world model step
                    one_hot = torch.zeros(1, action_dim, device=device)
                    one_hot[0, action] = 1.0
                    # Step world model (prior only)
                    gru_input = torch.cat([stoch, deter, one_hot], dim=-1)
                    deter = world_model.rssm.rssm_cell.gru(gru_input, deter)
                    prior_params = world_model.rssm.rssm_cell.prior_net(deter)
                    prior_mean, prior_logstd = prior_params.chunk(2, dim=-1)
                    prior_std = torch.exp(prior_logstd)
                    eps = torch.randn_like(prior_mean)
                    stoch = prior_mean + eps * prior_std
                    # Predict reward and value
                    reward = world_model.reward_pred(deter, stoch).squeeze()
                    value = critic(deter, stoch).squeeze()
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    values.append(value)
                    actions.append(one_hot)
                # Compute advantages (GAE)
                values_appended = values + [critic(deter, stoch).squeeze().detach()]
                advantages = compute_gae(rewards, values_appended, gamma=config.GAMMA, lam=0.95)
                batch_rollouts.append((log_probs, advantages, rewards, values, actions))
            # Update actor and critic
            actor_loss = 0.0
            critic_loss = 0.0
            for log_probs, advantages, rewards, values, actions in batch_rollouts:
                log_prob_sum = torch.stack(log_probs).sum()
                # Actor loss: -log_prob * advantage (with advantage detach)
                actor_loss += -(log_prob_sum * advantages.mean().detach())
                # Critic loss: MSE between predicted value and bootstrapped return
                returns = []
                R = 0
                for r in reversed(rewards):
                    R = r + config.GAMMA * R
                    returns.insert(0, R)
                returns = torch.tensor(returns, device=device)
                critic_loss += nn.MSELoss()(torch.stack(values), returns)
            actor_loss /= len(batch_rollouts)
            critic_loss /= len(batch_rollouts)
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()
            if (epoch+1) % 20 == 0:
                print(f"    Actor-critic epoch {epoch+1}/{config.ACTOR_EPOCHS}, actor loss: {actor_loss.item():.4f}, critic loss: {critic_loss.item():.4f}")

        # After training, get greedy action from actor on initial state
        with torch.no_grad():
            action_probs = actor(initial_deter, initial_stoch).cpu().numpy().flatten()
        sorted_idx = np.argsort(action_probs)[::-1]
        top_etfs = []
        for i in range(min(config.TOP_N, len(sorted_idx))):
            idx = sorted_idx[i]
            top_etfs.append({
                'ticker': tickers[idx],
                'weight': float(action_probs[idx])
            })
        print(f"  Top 3 ETFs by portfolio weight: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "run_date": today
        }

    # Save results
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/dreamer_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== DreamerV3 World Model Engine complete ===")

if __name__ == "__main__":
    main()
