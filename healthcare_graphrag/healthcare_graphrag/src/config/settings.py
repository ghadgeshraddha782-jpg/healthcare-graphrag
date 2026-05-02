"""
settings.py
Central configuration for the Healthcare GraphRAG System.
All secrets are loaded from environment variables — never hardcode them here.
"""

import os

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_PROVIDER  = os.environ.get("LLM_PROVIDER", "groq")   # "groq" | "openai"
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL     = os.environ.get("LLM_MODEL", "llama3-8b-8192")  # groq default
LLM_TEMP      = float(os.environ.get("LLM_TEMP", "0.3"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))

# ── TigerGraph ────────────────────────────────────────────────────────────────
TGRAPH_HOST     = os.environ.get("TGRAPH_HOST",     "https://your-host.i.tgcloud.io")
TGRAPH_GRAPH    = os.environ.get("TGRAPH_GRAPH",    "HealthcareGraph")
TGRAPH_USERNAME = os.environ.get("TGRAPH_USERNAME", "tigergraph")
TGRAPH_PASSWORD = os.environ.get("TGRAPH_PASSWORD", "")
TGRAPH_SECRET   = os.environ.get("TGRAPH_SECRET",   "")

# ── GraphRAG pipeline ─────────────────────────────────────────────────────────
GRAPH_RAG_TOP_K = int(os.environ.get("GRAPH_RAG_TOP_K", "5"))

# ── Data paths ────────────────────────────────────────────────────────────────
RAW_DATA_PATH       = os.path.join("data", "raw",       "dataset.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "symptom_disease_graph.csv")
