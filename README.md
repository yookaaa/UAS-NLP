# 🍳 AI Pencari Resep Masakan

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-FF6B35?style=for-the-badge)
![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-4CAF50?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-7B2FBE?style=for-the-badge)

**Sistem tanya-jawab resep masakan Indonesia berbasis RAG (Retrieval-Augmented Generation)**  
menggunakan LangChain · LangGraph · LangSmith · Ollama · ChromaDB

[Demo](#-demo--tampilan-aplikasi) · [Fitur](#-fitur-unggulan) · [Instalasi](#-instalasi--cara-menjalankan) · [Arsitektur](#-arsitektur-sistem)

</div>

---

## 📌 Deskripsi Proyek

**AI Pencari Resep Masakan** adalah sistem berbasis NLP/LLM yang memungkinkan pengguna untuk mencari dan menanyakan resep masakan Indonesia secara interaktif menggunakan bahasa alami. Sistem ini dibangun di atas tiga library utama:

| Library | Peran dalam Proyek |
|---|---|
| **LangChain** | Membangun RAG chain (retriever → prompt → LLM → output parser) menggunakan LCEL |
| **LangGraph** | Mengatur alur percakapan multi-node: klasifikasi query → retrieve & answer → format output |
| **LangSmith** | Monitoring & tracing seluruh langkah chain dan graph secara real-time |

Database pengetahuan berisi **54 resep masakan Nusantara** yang disimpan dalam vectorstore ChromaDB, sehingga LLM dapat memberikan jawaban yang akurat dan relevan berdasarkan konteks resep asli — bukan halusinasi.

---

## ✨ Fitur Unggulan

- 🤖 **RAG (Retrieval-Augmented Generation)** — LLM menjawab berdasarkan dokumen resep nyata
- 🧩 **LangGraph Multi-Node Workflow** — Alur kerja terstruktur dengan klasifikasi query otomatis
- 🔍 **Query Classification** — Sistem mendeteksi pertanyaan off-topic dan merespons dengan sopan
- 📊 **LangSmith Tracing** — Setiap langkah chain di-log untuk monitoring dan debugging
- 🏃 **100% Lokal** — Menggunakan Ollama, tidak perlu API key berbayar untuk LLM
- 💬 **Chat Interface** — UI interaktif seperti ChatGPT dengan riwayat percakapan
- 📚 **Badge Sumber Resep** — Menampilkan nama resep yang digunakan sebagai referensi jawaban
- 🍽️ **54 Resep Nusantara** — Dataset lengkap: Rendang, Soto, Bakso, Gado-Gado, dan masih banyak lagi

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB UI                          │
│              (src/app.py — Chat Interface)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │ query
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   LANGGRAPH WORKFLOW                         │
│                    (src/graph.py)                            │
│                                                              │
│   START                                                      │
│     │                                                        │
│     ▼                                                        │
│  [classify_query]  ←─── LLM (Ollama)                        │
│     │                                                        │
│   ┌─┴──────────────────────────────┐                        │
│   │ relevant          │ off_topic  │                        │
│   ▼                   ▼            │                        │
│  [retrieve_and_answer]  [handle_off_topic]                   │
│   │    ↑                   │                                 │
│   │    └── ChromaDB        │                                 │
│   │         Vectorstore    │                                 │
│   └──────────┬─────────────┘                                │
│              ▼                                               │
│         [format_output]                                      │
│              │                                               │
│             END                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  LANGCHAIN RAG CHAIN                         │
│                  (src/rag_chain.py)                          │
│                                                              │
│  query                                                       │
│    │                                                         │
│    ├──► Retriever (ChromaDB) ──► Top-3 Dokumen Resep        │
│    │                                   │                     │
│    └──────────────────────────────────►│                     │
│                                        ▼                     │
│                             ChatPromptTemplate               │
│                                        │                     │
│                                        ▼                     │
│                              ChatOllama (LLM)                │
│                                        │                     │
│                                        ▼                     │
│                              StrOutputParser                  │
│                                        │                     │
│                                     answer                   │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    LANGSMITH TRACING                         │
│          (otomatis bila LANGCHAIN_TRACING_V2=true)           │
│                                                              │
│  • Setiap node LangGraph di-trace                           │
│  • Setiap langkah RAG chain di-log                          │
│  • Waktu eksekusi & token usage tercatat                    │
└─────────────────────────────────────────────────────────────┘
```

### Komponen Teknis

| Komponen | Teknologi |
|---|---|
| LLM | Ollama (Qwen2.5, Gemma2, Llama3.2) |
| Embedding | `nomic-embed-text` via Ollama |
| Vector Database | ChromaDB (lokal) |
| RAG Framework | LangChain (LCEL) |
| Workflow Orchestration | LangGraph StateGraph |
| Observability | LangSmith |
| Web UI | Streamlit |

---

## 📂 Struktur Project

```
ai-resep-masakan/
│
├── 📁 data/
│   └── resep_masakan.csv          ← Dataset 54 resep masakan Indonesia
│
├── 📁 src/
│   ├── config.py                  ← Konfigurasi terpusat (env vars, paths)
│   ├── ingest.py                  ← Proses CSV → ChromaDB vectorstore
│   ├── rag_chain.py               ← LangChain RAG chain (LCEL)
│   ├── graph.py                   ← LangGraph multi-node workflow
│   └── app.py                     ← Streamlit web UI
│
├── 📁 vectorstore/                ← ChromaDB (dibuat saat ingest, di-gitignore)
├── 📁 docs/
│   └── screenshots/               ← Screenshot aplikasi
│
├── scripts_gen_dataset.py         ← Script pembuat dataset CSV
├── setup_check.py                 ← Cek semua dependensi sebelum jalan
├── requirements.txt
├── .env.example                   ← Template environment variable
├── .gitignore
└── README.md
```

---

## 📦 Dataset Resep

Dataset berisi **54 resep masakan Indonesia dan Nusantara** dengan atribut:

| Kolom | Keterangan |
|---|---|
| `id` | ID unik resep |
| `nama_resep` | Nama masakan |
| `kategori` | Jenis masakan (Makanan Utama, Sup, Camilan, dll.) |
| `asal_daerah` | Daerah asal masakan |
| `bahan` | Daftar bahan lengkap |
| `langkah` | Cara memasak step-by-step |
| `waktu_memasak_menit` | Estimasi waktu memasak |
| `porsi` | Jumlah porsi |
| `tingkat_kesulitan` | Mudah / Sedang / Sulit |
| `tag` | Label untuk filtering (pedas, vegetarian, dll.) |

**Contoh resep yang tersedia:** Rendang, Sate Ayam Madura, Gado-Gado, Soto Ayam Lamongan, Bakso Sapi, Nasi Goreng, Gudeg Jogja, Pempek Palembang, Rawon, Opor Ayam, dan 44 lainnya.

---

## 🔧 Cara Kerja LangChain (RAG Chain)

```python
# Pola LCEL (LangChain Expression Language) di rag_chain.py
rag_chain = (
    {
        "context": retriever | format_docs,   # Ambil dokumen → format teks
        "question": RunnablePassthrough(),     # Teruskan query asli
    }
    | prompt_template    # Masukkan ke prompt sistem
    | llm                # Kirim ke Ollama LLM
    | StrOutputParser()  # Parse output menjadi string
)
```

## 🧩 Cara Kerja LangGraph (State Machine)

```python
# StateGraph di graph.py
builder = StateGraph(RecipeState)
builder.add_node("classify_query", classify_query)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("handle_off_topic", handle_off_topic)
builder.add_node("format_output", format_output)

builder.add_edge(START, "classify_query")
builder.add_conditional_edges("classify_query", route_after_classify, {...})
builder.add_edge("retrieve_and_answer", "format_output")
builder.add_edge("handle_off_topic", "format_output")
builder.add_edge("format_output", END)
```

## 🔍 Cara Kerja LangSmith (Tracing)

```env
# Cukup set environment variable di .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_api_key_here
LANGCHAIN_PROJECT=ai-pencari-resep-masakan
```
LangChain dan LangGraph otomatis mengirim semua trace ke dashboard LangSmith tanpa perlu kode tambahan.

---

## 💻 Instalasi & Cara Menjalankan

### Prasyarat

- Python 3.10 atau lebih baru
- [Ollama](https://ollama.com/download) terinstal dan berjalan

### Langkah 1 — Clone Repository

```bash
git clone https://github.com/username/ai-resep-masakan.git
cd ai-resep-masakan
```

### Langkah 2 — Buat Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Langkah 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Langkah 4 — Pull Model Ollama

```bash
# Jalankan Ollama terlebih dahulu
ollama serve

# Di terminal baru, pull model yang dibutuhkan (pilih salah satu chat model)
ollama pull qwen2.5:7b        # Rekomendasi: cepat dan akurat (~4.7GB)
# ollama pull gemma2:9b       # Alternatif Google (~5.4GB)
# ollama pull llama3.2:3b     # Alternatif paling ringan (~2GB)

# Wajib: model embedding
ollama pull nomic-embed-text
```

### Langkah 5 — Konfigurasi Environment

```bash
cp .env.example .env
```

Edit file `.env` sesuai kebutuhan:
```env
OLLAMA_CHAT_MODEL=qwen2.5:7b        # Sesuaikan dengan model yang di-pull
OLLAMA_EMBED_MODEL=nomic-embed-text

# Untuk LangSmith (opsional tapi direkomendasikan):
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_api_key
```

### Langkah 6 — Cek Setup

```bash
python setup_check.py
```

### Langkah 7 — Generate Dataset

```bash
python scripts_gen_dataset.py
```

### Langkah 8 — Bangun Vectorstore

```bash
python src/ingest.py
```

> ⏳ Proses embedding ~54 dokumen membutuhkan waktu 2-5 menit tergantung spesifikasi komputer.

### Langkah 9 — Jalankan Aplikasi

```bash
streamlit run src/app.py
```

Buka browser dan akses: **http://localhost:8501**

---

## 🖼️ Demo & Tampilan Aplikasi

### Chat Interface Utama
Antarmuka chat interaktif dengan sidebar berisi contoh pertanyaan, info model, dan status LangSmith.

### Contoh Pertanyaan yang Bisa Ditanyakan

| Pertanyaan | Respon |
|---|---|
| *"Bagaimana cara membuat rendang?"* | Resep Rendang Daging Sapi lengkap dengan bahan & langkah |
| *"Carikan resep masakan pedas yang mudah"* | Beberapa rekomendasi masakan pedas + tingkat kesulitan |
| *"Bahan apa saja yang diperlukan untuk bakso?"* | Daftar bahan bakso sapi |
| *"Siapa presiden Indonesia?"* | Respons sopan bahwa pertanyaan di luar topik masakan |

---

## 🧪 Testing Individual Modul

```bash
# Test RAG chain saja
python src/rag_chain.py

# Test LangGraph workflow saja
python src/graph.py

# Cek vectorstore stats
python -c "
from src.config import VECTORSTORE_DIR
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
vs = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=OllamaEmbeddings(model='nomic-embed-text'), collection_name='resep_masakan')
print(f'Total dokumen: {vs._collection.count()}')
"
```

---

## ❓ Troubleshooting

| Masalah | Solusi |
|---|---|
| `Connection refused` saat ingest | Pastikan `ollama serve` berjalan di terminal terpisah |
| Model tidak ditemukan | Jalankan `ollama pull <nama-model>` |
| Vectorstore tidak ada | Jalankan `python src/ingest.py` |
| Jawaban tidak relevan | Coba ganti ke model yang lebih besar (qwen2.5:14b) |
| LangSmith tidak muncul | Cek API key di `.env` dan pastikan `LANGCHAIN_TRACING_V2=true` |

---

## 📚 Referensi

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Ollama Documentation](https://ollama.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 👤 Informasi Mahasiswa

> Diisi oleh mahasiswa sebelum pengumpulan UAS.

| | |
|---|---|
| **Nama** | Eka Maulana Hidayat |
| **NIM** | [NIM] |
| **Kelas** | [Kelas NLP] |
| **Mata Kuliah** | Natural Language Processing (NLP) |

---

<div align="center">

**🍳 Dibuat untuk UAS Mata Kuliah NLP**  
*Menggunakan LangChain · LangGraph · LangSmith · Ollama*