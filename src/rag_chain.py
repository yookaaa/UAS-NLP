"""
rag_chain.py
------------
Membangun RAG (Retrieval-Augmented Generation) chain menggunakan LangChain.

Alur kerja:
1. Retriever mengambil dokumen resep relevan dari ChromaDB berdasarkan query.
2. Prompt template menggabungkan konteks dokumen + pertanyaan user.
3. LLM Ollama (chat model) menghasilkan jawaban berdasarkan konteks.
4. Output parser memformat respons menjadi string bersih.

LangSmith otomatis men-trace seluruh chain ini jika
LANGCHAIN_TRACING_V2=true diset di .env.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_TEMPERATURE,
    RETRIEVER_TOP_K,
    VECTORSTORE_DIR,
)

# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Kamu adalah asisten memasak yang ramah dan berpengalaman \
bernama "ChefBot". Tugasmu membantu pengguna menemukan dan memahami resep \
masakan Indonesia dan Nusantara.

Gunakan HANYA informasi dari konteks resep yang diberikan di bawah ini untuk \
menjawab pertanyaan. Jika informasi tidak ada di konteks, katakan dengan jujur \
bahwa kamu tidak menemukan resep yang cocok dan sarankan pengguna untuk \
mencoba kata kunci lain.

Saat menjawab:
- Sampaikan nama resep dengan jelas.
- Sebutkan bahan-bahan secara terstruktur.
- Jelaskan langkah memasak dengan urutan yang mudah diikuti.
- Tambahkan info waktu memasak, porsi, dan tingkat kesulitan.
- Gunakan bahasa Indonesia yang ramah dan mudah dipahami.
- Jika ada lebih dari satu resep relevan, tampilkan keduanya.

Konteks Resep:
{context}
"""

HUMAN_PROMPT = "{question}"

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)


# ---------------------------------------------------------------------------
# Loader vectorstore
# ---------------------------------------------------------------------------
def load_vectorstore() -> Chroma:
    """Muat vectorstore ChromaDB yang sudah ada dari disk."""
    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"Vectorstore tidak ditemukan di: {VECTORSTORE_DIR}\n"
            "Jalankan dulu: python src/ingest.py"
        )

    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_EMBED_MODEL,
    )

    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
        collection_name="resep_masakan",
    )


# ---------------------------------------------------------------------------
# Builder retriever
# ---------------------------------------------------------------------------
def build_retriever(vectorstore: Chroma) -> BaseRetriever:
    """
    Buat retriever dari vectorstore.
    Menggunakan similarity search dengan top-k dokumen teratas.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_TOP_K},
    )


# ---------------------------------------------------------------------------
# Helper format dokumen
# ---------------------------------------------------------------------------
def _format_docs(docs: list) -> str:
    """Gabungkan beberapa Document menjadi satu string konteks."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"--- Resep {i} ---\n{doc.page_content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Builder RAG chain
# ---------------------------------------------------------------------------
def build_rag_chain(retriever: BaseRetriever) -> RunnableSerializable:
    """
    Bangun RAG chain dengan pola LCEL (LangChain Expression Language):

        query → retriever → format_docs
                         ↘
                           prompt → llm → output_parser
                         ↗
               passthrough (query)
    """
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_CHAT_MODEL,
        temperature=OLLAMA_TEMPERATURE,
    )

    rag_chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ---------------------------------------------------------------------------
# Fungsi helper publik (dipanggil oleh graph.py dan app.py)
# ---------------------------------------------------------------------------
def get_rag_chain() -> RunnableSerializable:
    """
    Load vectorstore, buat retriever, lalu kembalikan RAG chain.
    Dipakai oleh modul lain agar tidak perlu tahu detail inisialisasi.
    """
    vs = load_vectorstore()
    retriever = build_retriever(vs)
    return build_rag_chain(retriever)


def get_retriever() -> BaseRetriever:
    """Kembalikan retriever saja (tanpa chain), dipakai di graph node."""
    vs = load_vectorstore()
    return build_retriever(vs)


# ---------------------------------------------------------------------------
# Quick test (jalankan langsung: python src/rag_chain.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🔗 Menginisialisasi RAG chain ...")
    chain = get_rag_chain()
    test_query = "Bagaimana cara membuat rendang daging sapi?"
    print(f"\n❓ Pertanyaan: {test_query}\n")
    print("💬 Jawaban:")
    for chunk in chain.stream(test_query):
        print(chunk, end="", flush=True)
    print("\n")
