"""
app.py
------
Antarmuka pengguna (UI) berbasis Streamlit untuk AI Pencari Resep Masakan.

Cara menjalankan:
    streamlit run src/app.py

Fitur UI:
- Chat interface futuristik dengan glassmorphism
- Panel sidebar: info model Ollama + LangSmith status + contoh pertanyaan
- Streaming jawaban dari LLM
- Badge sumber resep yang ditemukan
- Tombol reset percakapan
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.config import (
    LANGSMITH_ENABLED,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    RETRIEVER_TOP_K,
    VECTORSTORE_DIR,
)
from src.graph import run_graph

# ---------------------------------------------------------------------------
# Konfigurasi halaman Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ChefBot — AI Dapur Masa Depan",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS futuristik + humanistik
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Reset & base ────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #070B14;
    }

    /* ── Animasi keyframes ───────────────────────── */
    @keyframes pulse-ring {
        0%   { box-shadow: 0 0 0 0 rgba(0, 212, 180, 0.5); }
        70%  { box-shadow: 0 0 0 14px rgba(0, 212, 180, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 212, 180, 0); }
    }

    @keyframes glow-flicker {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.82; }
    }

    @keyframes slide-in-up {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes scan-line {
        0%   { transform: translateY(-100%); }
        100% { transform: translateY(400%); }
    }

    @keyframes orbit {
        from { transform: rotate(0deg) translateX(22px) rotate(0deg); }
        to   { transform: rotate(360deg) translateX(22px) rotate(-360deg); }
    }

    /* ── Header hero ─────────────────────────────── */
    .hero-header {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #0D1B2E 0%, #0A1628 50%, #0D1520 100%);
        border: 1px solid rgba(0, 212, 180, 0.18);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.8rem;
        display: flex;
        align-items: center;
        gap: 2rem;
    }

    .hero-header::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse 60% 80% at 80% 50%, rgba(0, 212, 180, 0.06) 0%, transparent 70%),
                    radial-gradient(ellipse 40% 60% at 10% 80%, rgba(255, 140, 66, 0.05) 0%, transparent 60%);
        pointer-events: none;
    }

    /* scan-line dekoratif */
    .hero-header::after {
        content: '';
        position: absolute;
        left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 180, 0.4), transparent);
        animation: scan-line 4s ease-in-out infinite;
    }

    .hero-avatar-wrap {
        position: relative;
        flex-shrink: 0;
    }

    .hero-avatar {
        width: 72px;
        height: 72px;
        border-radius: 50%;
        background: linear-gradient(135deg, #00D4B4 0%, #00A896 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.2rem;
        animation: pulse-ring 2.4s ease-out infinite;
        position: relative;
        z-index: 2;
    }

    /* titik orbit */
    .hero-avatar-wrap::after {
        content: '●';
        font-size: 7px;
        color: #FF8C42;
        position: absolute;
        top: 50%; left: 50%;
        margin: -3.5px;
        animation: orbit 3s linear infinite;
    }

    .hero-text { flex: 1; }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0 0 0.3rem;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .hero-title span {
        background: linear-gradient(90deg, #00D4B4, #7DF9E4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-sub {
        font-size: 0.88rem;
        color: rgba(255,255,255,0.45);
        margin: 0;
        letter-spacing: 0.04em;
        font-weight: 300;
    }

    .hero-pills {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.8rem;
        flex-wrap: wrap;
    }

    .hero-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        border: 1px solid rgba(0, 212, 180, 0.3);
        color: #00D4B4;
        background: rgba(0, 212, 180, 0.07);
        letter-spacing: 0.05em;
    }

    /* ── Sidebar ──────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #060A12 !important;
        border-right: 1px solid rgba(0, 212, 180, 0.1) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.35) !important;
        margin-bottom: 0.8rem;
    }

    /* info box sidebar */
    section[data-testid="stSidebar"] .stAlert {
        background: rgba(0, 212, 180, 0.06) !important;
        border: 1px solid rgba(0, 212, 180, 0.2) !important;
        border-radius: 10px !important;
        color: rgba(255,255,255,0.75) !important;
        font-size: 0.82rem;
    }

    /* sidebar buttons */
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        width: 100%;
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        color: rgba(255,255,255,0.65) !important;
        font-size: 0.8rem !important;
        padding: 0.45rem 0.8rem !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
        font-family: 'Inter', sans-serif !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        background: rgba(0, 212, 180, 0.1) !important;
        border-color: rgba(0, 212, 180, 0.35) !important;
        color: #00D4B4 !important;
        transform: translateX(3px) !important;
    }

    /* reset button */
    section[data-testid="stSidebar"] div[data-testid="stButton"]:last-of-type > button {
        background: rgba(255, 140, 66, 0.08) !important;
        border-color: rgba(255, 140, 66, 0.25) !important;
        color: #FF8C42 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"]:last-of-type > button:hover {
        background: rgba(255, 140, 66, 0.18) !important;
        border-color: #FF8C42 !important;
    }

    /* sidebar divider */
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.07) !important;
    }

    /* sidebar caption */
    section[data-testid="stSidebar"] .stCaption {
        color: rgba(255,255,255,0.3) !important;
        font-size: 0.75rem !important;
    }

    /* ── Chat messages ────────────────────────────── */
    [data-testid="stChatMessage"] {
        animation: slide-in-up 0.3s ease both;
        padding: 0.9rem 1.1rem !important;
        border-radius: 14px !important;
        margin-bottom: 0.8rem !important;
    }

    /* Pesan assistant */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(0, 212, 180, 0.05) !important;
        border: 1px solid rgba(0, 212, 180, 0.12) !important;
    }

    /* Pesan user */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(255, 140, 66, 0.05) !important;
        border: 1px solid rgba(255, 140, 66, 0.12) !important;
    }

    [data-testid="stChatMessage"] p {
        color: rgba(255,255,255,0.88) !important;
        font-size: 0.93rem !important;
        line-height: 1.65 !important;
    }

    /* ── Source badges ────────────────────────────── */
    .source-wrap {
        margin-top: 0.6rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        align-items: center;
    }

    .source-label {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.3);
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.06em;
        margin-right: 0.2rem;
    }

    .source-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(0, 212, 180, 0.1);
        border: 1px solid rgba(0, 212, 180, 0.28);
        color: #00D4B4;
        padding: 0.22rem 0.65rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.01em;
    }

    /* ── Status indicators ───────────────────────── */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 0.4rem;
        vertical-align: middle;
    }

    .status-on  {
        background: #00D4B4;
        box-shadow: 0 0 8px #00D4B4;
        animation: glow-flicker 2s ease infinite;
    }

    .status-off { background: #3A3F4B; }

    .status-text-on  { color: #00D4B4; font-weight: 600; font-size: 0.85rem; }
    .status-text-off { color: #555B6A; font-weight: 600; font-size: 0.85rem; }

    /* ── Debug caption ───────────────────────────── */
    .debug-bar {
        margin-top: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.25);
        display: flex;
        gap: 1.2rem;
        flex-wrap: wrap;
    }

    .debug-bar span { display: flex; align-items: center; gap: 0.3rem; }
    .debug-bar b    { color: rgba(255,255,255,0.4); }

    /* ── Chat input ──────────────────────────────── */
    [data-testid="stChatInput"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(0, 212, 180, 0.2) !important;
        border-radius: 14px !important;
        color: white !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(0, 212, 180, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(0, 212, 180, 0.1) !important;
    }

    [data-testid="stChatInput"] textarea {
        color: rgba(255,255,255,0.88) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: rgba(255,255,255,0.25) !important;
    }

    /* ── Sidebar section title chip ─────────────── */
    .section-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.35);
        margin: 1rem 0 0.5rem;
    }

    .section-chip::before {
        content: '';
        display: block;
        width: 3px; height: 3px;
        border-radius: 50%;
        background: #00D4B4;
    }

    /* ── Error box ───────────────────────────────── */
    [data-testid="stAlert"] {
        background: rgba(255, 80, 80, 0.07) !important;
        border: 1px solid rgba(255, 80, 80, 0.25) !important;
        border-radius: 12px !important;
    }

    /* ── Spinner text ────────────────────────────── */
    .stSpinner > div {
        border-color: #00D4B4 transparent transparent !important;
    }

    /* ── Arsitektur code block ───────────────────── */
    .arch-block {
        background: rgba(0,0,0,0.4);
        border: 1px solid rgba(0, 212, 180, 0.12);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: rgba(255,255,255,0.45);
        line-height: 1.7;
        white-space: pre;
    }

    .arch-block .hl { color: #00D4B4; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-header">
        <div class="hero-avatar-wrap">
            <div class="hero-avatar">🍳</div>
        </div>
        <div class="hero-text">
            <h1 class="hero-title">Chef<span>Bot</span></h1>
            <p class="hero-sub">ASISTEN DAPUR BERBASIS KECERDASAN BUATAN</p>
            <div class="hero-pills">
                <span class="hero-pill">LangGraph</span>
                <span class="hero-pill">LangChain</span>
                <span class="hero-pill">Ollama</span>
                <span class="hero-pill">ChromaDB</span>
                <span class="hero-pill">RAG</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="section-chip">Konfigurasi Sistem</p>', unsafe_allow_html=True)

    st.markdown("**🤖 Model LLM**")
    st.info(f"**Chat:** `{OLLAMA_CHAT_MODEL}`\n\n**Embed:** `{OLLAMA_EMBED_MODEL}`")

    st.markdown("**📊 Pengaturan RAG**")
    st.info(f"**Top-K:** `{RETRIEVER_TOP_K}` dokumen\n\n**Store:** ChromaDB Lokal")

    st.markdown('<p class="section-chip">Observabilitas</p>', unsafe_allow_html=True)

    if LANGSMITH_ENABLED:
        st.markdown(
            '<p><span class="status-dot status-on"></span>'
            '<span class="status-text-on">LangSmith · AKTIF</span></p>',
            unsafe_allow_html=True,
        )
        st.caption("Semua sesi di-trace ke LangSmith dashboard.")
    else:
        st.markdown(
            '<p><span class="status-dot status-off"></span>'
            '<span class="status-text-off">LangSmith · NONAKTIF</span></p>',
            unsafe_allow_html=True,
        )
        st.caption("Tambahkan `LANGSMITH_API_KEY` di `.env` untuk mengaktifkan.")

    st.divider()

    st.markdown('<p class="section-chip">Coba Tanya Ini</p>', unsafe_allow_html=True)

    example_queries = [
        "Apa resep rendang daging sapi?",
        "Carikan resep masakan pedas yang mudah",
        "Bagaimana cara membuat soto ayam Lamongan?",
        "Resep camilan tradisional Jawa yang manis",
        "Masakan berbahan dasar tempe yang enak",
        "Bagaimana cara membuat bakso sapi?",
        "Resep sup yang cocok untuk cuaca dingin",
        "Masakan apa yang cepat dimasak dalam 20 menit?",
    ]

    for i, q in enumerate(example_queries):
        if st.button(q, key=f"btn_example_{i}", use_container_width=True):
            st.session_state["pending_query"] = q

    st.divider()

    if st.button("↺  Mulai Percakapan Baru", use_container_width=True, type="secondary"):
        st.session_state["messages"] = []
        st.session_state.pop("pending_query", None)
        st.rerun()

    st.divider()

    st.markdown('<p class="section-chip">Arsitektur</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="arch-block"><span class="hl">Query</span>
  │
  ▼
<span class="hl">[LangGraph]</span>
  ├─ classify_query
  ├─ retrieve_and_answer ←── ChromaDB
  └─ format_output
  │
  ▼
<span class="hl">Answer</span></div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Cek vectorstore
# ---------------------------------------------------------------------------
if not os.path.exists(VECTORSTORE_DIR):
    st.error(
        "**Vectorstore belum ditemukan.**\n\n"
        "Jalankan perintah berikut untuk membangun database resep:\n"
        "```bash\n"
        "python src/ingest.py\n"
        "```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Session state: riwayat pesan
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Halo! Saya **ChefBot** — asisten dapur AI yang siap memandu kamu "
                "menjelajahi kekayaan kuliner Indonesia.\n\n"
                "Tanya apa saja:\n"
                "- *Bagaimana cara membuat rendang yang empuk?*\n"
                "- *Carikan resep masakan pedas yang bisa selesai dalam 30 menit*\n"
                "- *Apa saja bahan untuk membuat bakso sapi dari nol?*\n\n"
                "Mau masak apa hari ini? 🍽️"
            ),
            "sources": [],
        }
    ]

# ---------------------------------------------------------------------------
# Tampilkan riwayat percakapan
# ---------------------------------------------------------------------------
for msg in st.session_state["messages"]:
    avatar = "🍳" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            badges = "".join(
                f'<span class="source-badge">📖 {s}</span>' for s in msg["sources"]
            )
            st.markdown(
                f'<div class="source-wrap">'
                f'<span class="source-label">SUMBER //</span>{badges}'
                f'</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Proses pending query dari tombol sidebar
# ---------------------------------------------------------------------------
pending = st.session_state.pop("pending_query", None)

# ---------------------------------------------------------------------------
# Input chat
# ---------------------------------------------------------------------------
user_input = st.chat_input("Tanya resep atau bahan masakan… ✦")

query_to_process = pending or user_input

if query_to_process:
    with st.chat_message("user", avatar="👤"):
        st.markdown(query_to_process)
    st.session_state["messages"].append(
        {"role": "user", "content": query_to_process, "sources": []}
    )

    with st.chat_message("assistant", avatar="🍳"):
        placeholder = st.empty()

        with st.spinner("Mencari & meracik jawaban terbaik…"):
            try:
                start = time.time()
                result = run_graph(query_to_process)
                elapsed = time.time() - start

                answer = result.get("answer", "Maaf, saya tidak bisa menghasilkan jawaban.")
                sources = result.get("sources", [])
                query_type = result.get("query_type", "unknown")

            except FileNotFoundError as e:
                answer = f"**Error:** {e}"
                sources = []
                query_type = "error"
                elapsed = 0.0

            except Exception as e:
                answer = (
                    f"**Terjadi kesalahan:** {e}\n\n"
                    "Pastikan Ollama sudah berjalan: `ollama serve`"
                )
                sources = []
                query_type = "error"
                elapsed = 0.0

        placeholder.markdown(answer)

        if sources:
            badges = "".join(
                f'<span class="source-badge">📖 {s}</span>' for s in sources
            )
            st.markdown(
                f'<div class="source-wrap">'
                f'<span class="source-label">SUMBER //</span>{badges}'
                f'</div>',
                unsafe_allow_html=True,
            )

        doc_count = len(result.get("retrieved_docs", []))
        st.markdown(
            f'<div class="debug-bar">'
            f'<span>⏱ <b>{elapsed:.2f}s</b></span>'
            f'<span>🏷 <b>{query_type}</b></span>'
            f'<span>📄 <b>{doc_count} dok</b></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )