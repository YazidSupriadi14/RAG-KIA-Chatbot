"""
RAG Buku KIA — versi CPU (GGUF via llama-cpp-python) untuk Hugging Face Spaces gratis.

Beda dari versi Colab (yang pakai transformers + bitsandbytes 4-bit, butuh GPU):
versi ini pakai model GGUF yang di-quantize khusus untuk jalan di CPU lewat llama-cpp-python.
Embedding & retrieval tetap sama persis (FAISS + sentence-transformers, sudah CPU-friendly
dari awal).

Struktur folder Space yang dibutuhkan:
    app.py                  <- file ini
    requirements.txt
    data/
        chunks.json         <- hasil dari notebook (Step 4b)
        faiss_index.bin     <- hasil dari notebook (Step 6)

Model GGUF di-download otomatis saat Space start (jangan commit file .gguf-nya sendiri ke
repo, biar Space-nya tetap ringan — cache HF yang nanganin).
"""

import json
import os

import faiss
import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
ARTIFACT_DIR = "./data"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"

# Model GGUF — cek dulu nama file persis di halaman HF repo-nya (bisa beda-beda
# tergantung siapa yang upload). Q4_K_M adalah titik seimbang antara ukuran & kualitas.
GGUF_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
GGUF_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"

MAX_CHARS_PER_CHUNK = 600
TOP_K = 5
N_CTX = 4096  # ukuran context window; turunkan kalau RAM Space kamu terbatas
N_THREADS = os.cpu_count() or 2

SYSTEM_PROMPT = """Kamu adalah asisten informasi kesehatan ibu hamil, bersalin, dan menyusui
berdasarkan Buku KIA (Kemenkes RI).

ATURAN KETAT:
1. Jawab HANYA berdasarkan konteks yang diberikan. Jangan menambahkan informasi dari luar konteks.
2. Jika konteks tidak cukup untuk menjawab, katakan dengan jujur bahwa informasi tidak tersedia
   dan sarankan konsultasi ke bidan/dokter.
3. Ini BUKAN pengganti konsultasi medis. Selalu sertakan pengingat ini bila relevan.
4. Jika ada tanda bahaya dalam pertanyaan atau konteks, WAJIB sarankan segera ke fasilitas
   kesehatan terdekat sebelum informasi lainnya.
"""

EXAMPLE_QUESTIONS = [
    "Apa tanda bahaya pada trimester 1?",
    "Bagaimana cara memerah dan menyimpan ASI?",
    "Berapa porsi makan ibu menyusui per hari?",
    "Anak saya batuk, bagaimana penanganannya?",
    "Aktivitas fisik apa yang tidak boleh dilakukan ibu hamil?",
]

# ---------------------------------------------------------------------------
# Load artifacts (sekali saja saat Space start)
# ---------------------------------------------------------------------------
print("Memuat chunks...")
with open(os.path.join(ARTIFACT_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

print("Memuat FAISS index...")
index = faiss.read_index(os.path.join(ARTIFACT_DIR, "faiss_index.bin"))

print("Memuat model embedding (CPU)...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

print(f"Mengunduh & memuat model GGUF ({GGUF_REPO}/{GGUF_FILENAME})...")
gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILENAME)
llm = Llama(
    model_path=gguf_path,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    chat_format="chatml",  # Qwen2.5 pakai format ChatML
    verbose=False,
)
print("Model siap. n_threads:", N_THREADS)


# ---------------------------------------------------------------------------
# Retrieval (identik dengan versi Colab)
# ---------------------------------------------------------------------------
def keyword_boost(query, chunk_text, chunk_label=None):
    query_words = set(w.lower() for w in query.split() if len(w) > 3)
    text_lower = (chunk_text + " " + (chunk_label or "")).lower()
    matches = sum(1 for w in query_words if w in text_lower)
    return matches * 0.05


def retrieve(query, top_k=TOP_K, initial_k=15):
    query_vec = embed_model.encode(["query: " + query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, initial_k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = all_chunks[idx]
        boost = keyword_boost(query, chunk["text"], chunk.get("item_label"))
        candidates.append({**chunk, "score": float(score) + boost})

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Generation (GGUF via llama-cpp-python, bukan transformers lagi)
# ---------------------------------------------------------------------------
def generate_answer(query, retrieved_chunks, max_tokens=800):
    context = "\n\n".join(
        f"[{c['section_title']} - p.{c['pdf_page']}] {c['text'][:MAX_CHARS_PER_CHUNK]}"
        for c in retrieved_chunks
    )
    has_danger = any(c["is_danger_sign"] for c in retrieved_chunks)
    danger_note = (
        "\n\nPERHATIAN: Konteks ini mengandung informasi tanda bahaya. "
        "Pastikan jawabanmu menyertakan arahan untuk segera ke fasilitas kesehatan."
        if has_danger
        else ""
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + danger_note},
        {"role": "user", "content": f"Konteks:\n{context}\n\nPertanyaan: {query}"},
    ]

    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
        repeat_penalty=1.1,
    )
    answer = result["choices"][0]["message"]["content"].strip()
    return answer, has_danger


# ---------------------------------------------------------------------------
# UI helpers & chat function
# ---------------------------------------------------------------------------
def format_sources(retrieved_chunks):
    seen = []
    for c in retrieved_chunks:
        label = f"{c['section_title']} \u00b7 hal. {c['pdf_page']}"
        if label not in seen:
            seen.append(label)
    return seen


def chat_fn(message, history):
    retrieved = retrieve(message)
    answer, has_danger = generate_answer(message, retrieved)
    sources = format_sources(retrieved)

    parts = [answer.strip()]

    if has_danger:
        parts.append(
            "\n\n> \u26a0\ufe0f **Kalau ini kondisi darurat, segera hubungi bidan/dokter "
            "atau ke fasilitas kesehatan terdekat.**"
        )

    if sources:
        sources_text = "  \n".join(f"\U0001F4C4 *{s}*" for s in sources)
        parts.append(f"\n\n---\n**Sumber:**  \n{sources_text}")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Tema & layout
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.pink,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Poppins"), "sans-serif"],
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    block_radius="16px",
    block_shadow="0 2px 12px rgba(0,0,0,0.06)",
)

CUSTOM_CSS = """
#header-box {
    background: linear-gradient(135deg, #fbcfe8 0%, #a7f3d0 100%);
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 12px;
}
#header-box h1 { margin: 0 0 4px 0; font-size: 26px; color: #831843; }
#header-box p { margin: 0; color: #065f46; font-size: 14px; }
#disclaimer { font-size: 12px; color: #64748b; text-align: center; margin-top: 8px; }
"""

with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="RAG Buku KIA") as demo:
    gr.HTML(
        """
        <div id="header-box">
            <h1>\U0001F931 Tanya Buku KIA</h1>
            <p>Asisten seputar kehamilan, persalinan, dan menyusui \u2014 berdasarkan
            Buku Kesehatan Ibu dan Anak (Kemenkes RI). Bukan pengganti konsultasi medis.</p>
        </div>
        """
    )

    gr.ChatInterface(
        fn=chat_fn,
        examples=EXAMPLE_QUESTIONS,
        chatbot=gr.Chatbot(height=480, avatar_images=(None, "\U0001F931")),
        textbox=gr.Textbox(
            placeholder="Tulis pertanyaan seputar kehamilan, persalinan, atau menyusui..."
        ),
    )

    gr.HTML(
        '<div id="disclaimer">\u2695\ufe0f Informasi ini bersifat umum dan tidak menggantikan '
        "pemeriksaan langsung oleh tenaga kesehatan.</div>"
    )

if __name__ == "__main__":
    demo.launch()
