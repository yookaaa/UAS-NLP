"""
config.py
---------
Konfigurasi terpusat untuk project AI Pencari Resep Masakan.

Semua environment variable dibaca di sini supaya modul lain tidak perlu
mengurus `os.getenv` berulang-ulang. Nilai default mengarah ke setup Ollama
lokal yang paling umum dipakai.
"""
import os
from dotenv import load_dotenv

# Muat file .env (jika ada) ke environment variable Python.
load_dotenv()

# ---------------------------------------------------------------------------
# OLLAMA (LLM lokal)
# ---------------------------------------------------------------------------
# Model chat lokal yang dijalankan via Ollama. Ganti sesuai model yang sudah
# di-`ollama pull` di komputer kamu, contoh: "qwen2.5:7b", "gemma2:9b",
# "llama3.2:3b".
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))

# ---------------------------------------------------------------------------
# LANGSMITH (observability / tracing)
# ---------------------------------------------------------------------------
# LangChain & LangGraph otomatis mengirim trace ke LangSmith ketika
# LANGCHAIN_TRACING_V2=true dan LANGCHAIN_API_KEY terisi. Tidak perlu kode
# tambahan apa pun di luar set environment variable ini.
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_ENDPOINT", os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"))
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY"))
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "ai-pencari-resep-masakan"))

LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

# ---------------------------------------------------------------------------
# PATH & DATA
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.getenv("DATA_PATH", os.path.join(BASE_DIR, "data", "resep_masakan.csv"))
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", os.path.join(BASE_DIR, "vectorstore"))

# Jumlah dokumen relevan yang diambil retriever untuk setiap pertanyaan.
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "3"))
