"""
ingest.py
---------
Membaca dataset resep masakan dari CSV, mengubahnya menjadi Document LangChain,
lalu membangun vectorstore ChromaDB menggunakan embedding Ollama (nomic-embed-text).

Jalankan sekali sebelum memakai aplikasi:
    python src/ingest.py
"""
import csv
import os
import sys

# Tambahkan root project ke sys.path agar import src.config bisa berjalan.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.config import (
    DATA_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    VECTORSTORE_DIR,
)


# ---------------------------------------------------------------------------
# 1. Baca CSV dan bangun list Document
# ---------------------------------------------------------------------------
def load_documents(csv_path: str) -> list[Document]:
    """
    Baca setiap baris CSV resep dan ubah menjadi Document LangChain.

    Konten dokumen (page_content) digabung dalam format teks narasi agar
    lebih mudah diproses oleh embedding model. Metadata (kategori, asal
    daerah, dll.) disimpan terpisah supaya bisa dipakai filter retrieval.
    """
    documents: list[Document] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Format page_content sebagai teks deskriptif agar hasil embedding
            # lebih kontekstual dan retrieval lebih akurat.
            page_content = (
                f"Nama Resep: {row['nama_resep']}\n"
                f"Kategori: {row['kategori']}\n"
                f"Asal Daerah: {row['asal_daerah']}\n"
                f"Tingkat Kesulitan: {row['tingkat_kesulitan']}\n"
                f"Waktu Memasak: {row['waktu_memasak_menit']} menit\n"
                f"Porsi: {row['porsi']} orang\n"
                f"Bahan-Bahan:\n{row['bahan'].replace(';', chr(10) + '-')}\n"
                f"Cara Memasak:\n{row['langkah']}\n"
                f"Tag: {row['tag'].replace(';', ', ')}"
            )

            metadata = {
                "id": int(row["id"]),
                "nama_resep": row["nama_resep"],
                "kategori": row["kategori"],
                "asal_daerah": row["asal_daerah"],
                "tingkat_kesulitan": row["tingkat_kesulitan"],
                "waktu_memasak_menit": int(row["waktu_memasak_menit"]),
                "porsi": int(row["porsi"]),
                "tag": row["tag"],
            }

            documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


# ---------------------------------------------------------------------------
# 2. Bangun vectorstore dari list Document
# ---------------------------------------------------------------------------
def build_vectorstore(documents: list[Document]) -> Chroma:
    """
    Embed semua dokumen dengan Ollama dan simpan hasilnya ke ChromaDB lokal.
    Jika vectorstore sudah ada sebelumnya, data lama akan ditimpa.
    """
    print(f"🔧 Menggunakan embedding model  : {OLLAMA_EMBED_MODEL}")
    print(f"🔧 Target vectorstore           : {VECTORSTORE_DIR}")
    print(f"📄 Jumlah dokumen resep          : {len(documents)}")

    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_EMBED_MODEL,
    )

    # Hapus vectorstore lama agar tidak tercampur data sebelumnya.
    if os.path.exists(VECTORSTORE_DIR):
        import shutil
        shutil.rmtree(VECTORSTORE_DIR)
        print("🗑️  Vectorstore lama dihapus.")

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    print("⏳ Membuat embedding ... (membutuhkan beberapa menit)")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
        collection_name="resep_masakan",
    )

    print(f"✅ Vectorstore berhasil dibuat! Total koleksi: {vectorstore._collection.count()} item.")
    return vectorstore


# ---------------------------------------------------------------------------
# 3. Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print(" INGEST: Membangun Vector Store Resep Masakan")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset tidak ditemukan di: {DATA_PATH}")
        print("   Jalankan dulu: python scripts_gen_dataset.py")
        sys.exit(1)

    docs = load_documents(DATA_PATH)
    build_vectorstore(docs)
    print("\n🎉 Proses ingest selesai! Siap digunakan.\n")


if __name__ == "__main__":
    main()
