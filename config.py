import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-dreamer-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Training data window (days) for world model
TRAIN_WINDOW = 252
# Latent state dimensions
RSSM_DETERMINISTIC = 64
RSSM_STOCHASTIC = 32
RSSM_HIDDEN = 64

# World model training
WM_BATCH_SIZE = 32
WM_LEARNING_RATE = 1e-3
WM_EPOCHS = 50

# Actor-critic training on imagined rollouts
ROLLOUT_LENGTH = 20
ACTOR_HIDDEN = 64
ACTOR_LEARNING_RATE = 1e-4
ACTOR_EPOCHS = 100
ACTOR_BATCH_SIZE = 32

# Risk-free rate for portfolio returns (daily)
RISK_FREE_RATE = 0.0

TOP_N = 3   # Number of ETFs to output (will show top weights)
