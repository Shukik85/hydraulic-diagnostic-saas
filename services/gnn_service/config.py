import os
from dotenv import load_dotenv

load_dotenv()


# CUDA 12.9
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
TORCH_CUDA_ARCH_LIST="7.5"  # GTX 1650 compute capability

# Device
DEVICE = os.getenv("DEVICE", "cuda")

# Model
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "96"))
NUM_GAT_LAYERS = int(os.getenv("NUM_GAT_LAYERS", "3"))
NUM_HEADS = int(os.getenv("NUM_HEADS", "4"))
LSTM_LAYERS = int(os.getenv("LSTM_LAYERS", "2"))
DROPOUT = float(os.getenv("DROPOUT", "0.12"))

# Temporal
TIME_WINDOW_MINUTES = int(os.getenv("TIME_WINDOW_MINUTES", "60"))
TIMESTEP_MINUTES = int(os.getenv("TIMESTEP_MINUTES", "5"))
NUM_TIMESTEPS = TIME_WINDOW_MINUTES // TIMESTEP_MINUTES

# Thresholds
HEALTH_CRITICAL = float(os.getenv("HEALTH_CRITICAL", "0.3"))
HEALTH_WARNING = float(os.getenv("HEALTH_WARNING", "0.6"))

# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "hydraulic_db")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))

# LLM/RAG
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
