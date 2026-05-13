# DreamerV3 World Model Engine

**Model‑based reinforcement learning** for ETF portfolio selection.  
Learns a latent dynamics model (RSSM) of asset returns and trains an actor‑critic purely on imagined rollouts (20 steps) in the latent space. Outputs portfolio weights (long only) for all ETFs in the universe.

- **World model:** RSSM with GRU, latent dim 64+32, trained on 252‑day sequences
- **Actor‑critic:** trained on 20‑step imagined futures, discount 0.99
- **Output:** top 3 ETFs by weight (highest allocation)
- Runs daily on GitHub Actions (approx 2‑3 hours)

## Run locally
```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
