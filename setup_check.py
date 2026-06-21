"""
setup_check.py
--------------
Script diagnostik untuk memverifikasi semua kebutuhan project sudah siap.

Jalankan sebelum ingest/app:
    python setup_check.py

Cek yang dilakukan:
1. Python version >= 3.10
2. Semua package yang dibutuhkan terinstal
3. Ollama service bisa diakses
4. Model chat & embedding tersedia di Ollama
5. Dataset CSV tersedia
6. Vectorstore sudah dibangun (opsional, ditandai warning)
"""
import importlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REQUIRED_PACKAGES = {
    "langchain": "langchain",
    "langchain_community": "langchain-community",
    "langchain_chroma": "langchain-chroma",
    "langchain_ollama": "langchain-ollama",
    "langgraph": "langgraph",
    "langsmith": "langsmith",
    "streamlit": "streamlit",
    "chromadb": "chromadb",
    "dotenv": "python-dotenv",
}

OK = "✅"
WARN = "⚠️ "
ERR = "❌"


def check_python_version() -> bool:
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    status = OK if ok else ERR
    print(f"{status} Python {v.major}.{v.minor}.{v.micro}", end="")
    if not ok:
        print(" (butuh >= 3.10)")
    else:
        print()
    return ok


def check_packages() -> bool:
    all_ok = True
    for module, pkg in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module)
            print(f"{OK} {pkg}")
        except ImportError:
            print(f"{ERR} {pkg}  →  pip install {pkg}")
            all_ok = False
    return all_ok


def check_ollama_service() -> bool:
    try:
        import urllib.request
        from src.config import OLLAMA_BASE_URL
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        print(f"{OK} Ollama service berjalan di {OLLAMA_BASE_URL}")
        return True
    except Exception as e:
        print(f"{ERR} Ollama tidak bisa diakses: {e}")
        print("   → Jalankan: ollama serve")
        return False


def check_ollama_models() -> bool:
    try:
        import json
        import urllib.request
        from src.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL

        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as r:
            data = json.loads(r.read())

        installed = [m["name"] for m in data.get("models", [])]
        all_ok = True

        for model in [OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL]:
            # Cek apakah ada nama model yang cocok (prefix match)
            found = any(m.startswith(model.split(":")[0]) for m in installed)
            if found:
                print(f"{OK} Ollama model: {model}")
            else:
                print(f"{WARN} Ollama model '{model}' belum di-pull.")
                print(f"   → Jalankan: ollama pull {model}")
                all_ok = False

        return all_ok
    except Exception as e:
        print(f"{WARN} Tidak bisa cek model Ollama: {e}")
        return False


def check_dataset() -> bool:
    from src.config import DATA_PATH
    if os.path.exists(DATA_PATH):
        import csv
        with open(DATA_PATH, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"{OK} Dataset CSV: {len(rows)} resep ditemukan  ({DATA_PATH})")
        return True
    else:
        print(f"{ERR} Dataset CSV tidak ditemukan: {DATA_PATH}")
        print("   → Jalankan: python scripts_gen_dataset.py")
        return False


def check_vectorstore() -> bool:
    from src.config import VECTORSTORE_DIR
    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        print(f"{OK} Vectorstore: ditemukan di {VECTORSTORE_DIR}")
        return True
    else:
        print(f"{WARN} Vectorstore belum dibangun.")
        print("   → Jalankan: python src/ingest.py")
        return False  # hanya warning, bukan error fatal


def main():
    print("=" * 60)
    print(" SETUP CHECK: AI Pencari Resep Masakan")
    print("=" * 60)

    results = {}

    print("\n[1] Python Version")
    results["python"] = check_python_version()

    print("\n[2] Packages Python")
    results["packages"] = check_packages()

    print("\n[3] Ollama Service")
    results["ollama_service"] = check_ollama_service()

    if results["ollama_service"]:
        print("\n[4] Ollama Models")
        results["ollama_models"] = check_ollama_models()
    else:
        results["ollama_models"] = False

    print("\n[5] Dataset CSV")
    results["dataset"] = check_dataset()

    print("\n[6] Vectorstore")
    check_vectorstore()

    # Ringkasan
    print("\n" + "=" * 60)
    critical_ok = all(results[k] for k in ["python", "packages", "dataset"])
    if critical_ok:
        print("🎉 Semua komponen kritis OK!")
        print("\nLangkah selanjutnya:")
        print("  1. Jika belum: python src/ingest.py   ← bangun vectorstore")
        print("  2. Jalankan app: streamlit run src/app.py")
    else:
        print("⚠️  Ada komponen yang bermasalah. Perbaiki dulu sebelum lanjut.")
    print("=" * 60)


if __name__ == "__main__":
    main()
