"""
graph.py
--------
Mendefinisikan LangGraph StateGraph untuk sistem pencari resep masakan.

Alur (graph flow):
    START
      │
      ▼
  [classify_query]  ← Cek apakah pertanyaan relevan dengan resep
      │
    ┌─┴──────────────────────────┐
    │ relevant                   │ off_topic
    ▼                            ▼
  [retrieve_and_answer]    [handle_off_topic]
    │                            │
    └──────────┬─────────────────┘
               ▼
           [format_output]
               │
               ▼
             END

Node-node:
- classify_query      : LLM menentukan apakah query berkaitan dengan resep/masakan.
- retrieve_and_answer : RAG chain mengambil dokumen & menghasilkan jawaban.
- handle_off_topic    : Respons sopan untuk pertanyaan di luar topik.
- format_output       : Menambahkan metadata (sumber resep) ke output akhir.

LangSmith akan otomatis men-trace seluruh langkah graph ini jika
LANGCHAIN_TRACING_V2=true.
"""
import os
import sys
from typing import Annotated, Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_TEMPERATURE,
)
from src.rag_chain import _format_docs, build_rag_chain, get_retriever

# ---------------------------------------------------------------------------
# State: skema data yang mengalir antar node
# ---------------------------------------------------------------------------
class RecipeState(TypedDict):
    """
    State yang dibawa sepanjang eksekusi graph.

    Atribut:
        query         : Pertanyaan asli dari pengguna.
        query_type    : Hasil klasifikasi: 'relevant' atau 'off_topic'.
        retrieved_docs: Dokumen resep yang berhasil diambil oleh retriever.
        answer        : Jawaban final yang akan ditampilkan ke pengguna.
        sources       : Daftar nama resep yang digunakan sebagai sumber.
    """
    query: str
    query_type: str
    retrieved_docs: list[Document]
    answer: str
    sources: list[str]


# ---------------------------------------------------------------------------
# LLM instance (dipakai bersama di semua node)
# ---------------------------------------------------------------------------
_llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_CHAT_MODEL,
    temperature=OLLAMA_TEMPERATURE,
)

# Retriever (diinisialisasi sekali, dipakai oleh node retrieve_and_answer)
_retriever = None
_rag_chain = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


def _get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = build_rag_chain(_get_retriever())
    return _rag_chain


# ---------------------------------------------------------------------------
# Node 1: classify_query
# ---------------------------------------------------------------------------
def classify_query(state: RecipeState) -> RecipeState:
    """
    Klasifikasikan apakah pertanyaan user berkaitan dengan resep / memasak.

    Menggunakan LLM dengan prompt zero-shot sederhana dan meminta output
    hanya berupa satu kata: 'relevant' atau 'off_topic'.
    """
    classification_prompt = f"""Kamu adalah sistem klasifikasi pertanyaan.
Tugasmu HANYA menentukan apakah pertanyaan berikut berkaitan dengan:
- Resep masakan
- Cara memasak / mengolah bahan makanan
- Informasi bahan / bumbu makanan
- Rekomendasi makanan / minuman
- Teknik memasak

Jika ya, jawab hanya dengan kata: relevant
Jika tidak, jawab hanya dengan kata: off_topic

Pertanyaan: {state['query']}

Jawaban (relevant/off_topic):"""

    response = _llm.invoke(classification_prompt)
    raw = response.content.strip().lower()

    # Normalisasi: jika tidak persis "relevant" atau "off_topic", coba tebak.
    if "relevant" in raw:
        query_type = "relevant"
    else:
        query_type = "off_topic"

    return {**state, "query_type": query_type}


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: RecipeState) -> RecipeState:
    """
    Ambil dokumen resep relevan dari vectorstore lalu hasilkan jawaban
    menggunakan RAG chain.
    """
    query = state["query"]

    # Retrieve dokumen terlebih dahulu (untuk metadata sumber).
    retriever = _get_retriever()
    docs: list[Document] = retriever.invoke(query)

    # Hasilkan jawaban via RAG chain.
    rag_chain = _get_rag_chain()
    answer = rag_chain.invoke(query)

    # Kumpulkan nama resep sebagai sumber.
    sources = list({doc.metadata.get("nama_resep", "Tidak diketahui") for doc in docs})

    return {**state, "retrieved_docs": docs, "answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# Node 3: handle_off_topic
# ---------------------------------------------------------------------------
def handle_off_topic(state: RecipeState) -> RecipeState:
    """
    Beri respons sopan untuk pertanyaan yang tidak berhubungan dengan masakan.
    """
    off_topic_response = (
        "Halo! 👋 Saya **ChefBot**, asisten memasak virtual yang fokus pada "
        "resep dan kuliner Indonesia.\n\n"
        "Sepertinya pertanyaan kamu tidak berkaitan dengan memasak atau resep. "
        "Saya hanya bisa membantu untuk hal-hal seperti:\n"
        "- 🍜 Mencari resep masakan tertentu\n"
        "- 🌶️ Menanyakan bahan-bahan yang diperlukan\n"
        "- 🔥 Cara memasak atau teknik memasak\n"
        "- 🍱 Rekomendasi masakan berdasarkan selera\n\n"
        "Silakan coba tanyakan sesuatu tentang masakan, ya! 😊"
    )

    return {**state, "retrieved_docs": [], "answer": off_topic_response, "sources": []}


# ---------------------------------------------------------------------------
# Node 4: format_output
# ---------------------------------------------------------------------------
def format_output(state: RecipeState) -> RecipeState:
    """
    Tambahkan footer sumber resep ke jawaban (jika ada).
    """
    if state["sources"]:
        sources_text = "\n\n---\n📚 **Sumber resep:** " + " | ".join(state["sources"])
        answer_with_sources = state["answer"] + sources_text
    else:
        answer_with_sources = state["answer"]

    return {**state, "answer": answer_with_sources}


# ---------------------------------------------------------------------------
# Routing condition
# ---------------------------------------------------------------------------
def route_after_classify(state: RecipeState) -> Literal["retrieve_and_answer", "handle_off_topic"]:
    """
    Tentukan node berikutnya berdasarkan hasil klasifikasi query.
    Fungsi ini dipakai sebagai conditional edge di StateGraph.
    """
    if state["query_type"] == "relevant":
        return "retrieve_and_answer"
    return "handle_off_topic"


# ---------------------------------------------------------------------------
# Builder graph
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    """
    Rakit semua node dan edge menjadi StateGraph siap dijalankan.

    Graph dikompilasi sekali (compile) agar bisa di-invoke berkali-kali
    tanpa harus membangun ulang strukturnya.
    """
    builder = StateGraph(RecipeState)

    # Daftarkan semua node.
    builder.add_node("classify_query", classify_query)
    builder.add_node("retrieve_and_answer", retrieve_and_answer)
    builder.add_node("handle_off_topic", handle_off_topic)
    builder.add_node("format_output", format_output)

    # Tentukan edge.
    builder.add_edge(START, "classify_query")

    # Conditional edge setelah klasifikasi.
    builder.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "handle_off_topic": "handle_off_topic",
        },
    )

    # Kedua jalur bertemu di format_output lalu END.
    builder.add_edge("retrieve_and_answer", "format_output")
    builder.add_edge("handle_off_topic", "format_output")
    builder.add_edge("format_output", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Helper publik
# ---------------------------------------------------------------------------
def run_graph(query: str) -> dict:
    """
    Jalankan graph dengan sebuah query dan kembalikan state akhir.

    Returns:
        dict berisi 'answer', 'sources', 'query_type', dan 'retrieved_docs'.
    """
    graph = build_graph()
    initial_state: RecipeState = {
        "query": query,
        "query_type": "",
        "retrieved_docs": [],
        "answer": "",
        "sources": [],
    }
    result = graph.invoke(initial_state)
    return result


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🧩 Testing LangGraph workflow ...\n")

    test_cases = [
        "Apa resep soto ayam yang enak?",
        "Bagaimana cara main gitar?",   # off-topic
        "Carikan resep masakan pedas yang cepat dibuat",
    ]

    for q in test_cases:
        print(f"❓ Query  : {q}")
        out = run_graph(q)
        print(f"🏷️  Tipe   : {out['query_type']}")
        print(f"📚 Sumber : {', '.join(out['sources']) if out['sources'] else '-'}")
        print(f"💬 Jawaban:\n{out['answer'][:300]}...")
        print("-" * 60)
