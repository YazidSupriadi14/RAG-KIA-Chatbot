"""
RAG Buku KIA — versi Streamlit untuk Streamlit Community Cloud.

Backend (retrieval + generation) identik dengan versi CPU/GGUF sebelumnya:
FAISS + sentence-transformers untuk retrieval, llama-cpp-python (GGUF) untuk generation.
Cuma lapisan UI-nya yang diganti dari Gradio ke Streamlit.

Struktur folder repo GitHub yang dibutuhkan (Streamlit Cloud connect langsung ke GitHub):
    streamlit_app.py        <- file ini (atau app.py, sesuaikan saat setup di Streamlit Cloud)
    requirements.txt
    data/
        chunks.json          <- hasil dari notebook (Step 4b)
        faiss_index.bin      <- hasil dari notebook (Step 6)

Cara deploy:
1. Push semua file di atas ke repo GitHub (public atau private).
2. Buka share.streamlit.io, login pakai GitHub, klik "New app".
3. Pilih repo, branch, dan file ini sebagai main file.
4. Deploy — boot pertama agak lama (download model GGUF ~2GB).
"""

import json
import os

import faiss
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
ARTIFACT_DIR = "./data"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"

GGUF_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
GGUF_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"  # cek nama persis di halaman HF repo-nya

MAX_CHARS_PER_CHUNK = 600
TOP_K = 5
N_CTX = 4096

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
# Load artifacts & model — pakai st.cache_resource biar cuma jalan SEKALI,
# bukan tiap kali ada interaksi (ini poin penting yang sempat dibahas soal
# risiko Streamlit reload berulang kalau lupa caching)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat data & model (cuma sekali di awal)...")
def load_pipeline():
    with open(os.path.join(ARTIFACT_DIR, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    idx = faiss.read_index(os.path.join(ARTIFACT_DIR, "faiss_index.bin"))
    embed = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

    gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILENAME)
    llm = Llama(
        model_path=gguf_path,
        n_ctx=N_CTX,
        n_threads=os.cpu_count() or 2,
        chat_format="chatml",
        verbose=False,
    )
    return chunks, idx, embed, llm


all_chunks, index, embed_model, llm = load_pipeline()


# ---------------------------------------------------------------------------
# Retrieval & Generation
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


def format_sources(retrieved_chunks):
    seen = []
    for c in retrieved_chunks:
        label = f"{c['section_title']} \u00b7 hal. {c['pdf_page']}"
        if label not in seen:
            seen.append(label)
    return seen


def answer_question(query):
    retrieved = retrieve(query)
    answer, has_danger = generate_answer(query, retrieved)
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
# UI — Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG Buku KIA", page_icon="\U0001F931", layout="centered")

st.markdown(
    """
    <style>
    .header-box {
        background: linear-gradient(135deg, #fbcfe8 0%, #a7f3d0 100%);
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 20px;
    }
    .header-box h1 { margin: 0 0 4px 0; font-size: 26px; color: #831843; }
    .header-box p { margin: 0; color: #065f46; font-size: 14px; }
    .disclaimer { font-size: 12px; color: #64748b; text-align: center; margin-top: 16px; }
    </style>
    <div class="header-box">
        <h1>\U0001F931 Tanya Buku KIA</h1>
        <p>Asisten seputar kehamilan, persalinan, dan menyusui \u2014 berdasarkan
        Buku Kesehatan Ibu dan Anak (Kemenkes RI). Bukan pengganti konsultasi medis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Contoh pertanyaan (cuma tampil kalau belum ada percakapan)
if not st.session_state.messages:
    st.markdown("**\U0001F4A1 Contoh pertanyaan:**")
    cols = st.columns(1)
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.spinner("Mencari jawaban..."):
                answer = answer_question(q)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    avatar = "\U0001F931" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Input pertanyaan baru
if prompt := st.chat_input("Tulis pertanyaan seputar kehamilan, persalinan, atau menyusui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="\U0001F931"):
        with st.spinner("Mencari jawaban..."):
            answer = answer_question(prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown(
    '<div class="disclaimer">\u2695\ufe0f Informasi ini bersifat umum dan tidak menggantikan '
    "pemeriksaan langsung oleh tenaga kesehatan.</div>",
    unsafe_allow_html=True,
)