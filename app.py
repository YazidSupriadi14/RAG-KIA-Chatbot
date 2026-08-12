"""
RAG Buku KIA — versi ZeroGPU untuk Hugging Face Spaces.

ZeroGPU cuma minjemin GPU pas fungsi yang didekorasi @spaces.GPU dipanggil — jadi
model di-load sekali di awal (di luar fungsi), tapi actual inference (model.generate)
dibungkus @spaces.GPU supaya dapat akses GPU cuma pas dibutuhkan.

PENTING:
- Space SDK harus Gradio (ZeroGPU cuma kompatibel dengan Gradio SDK).
- Space hardware di Settings harus dipilih "ZeroGPU".
- Kuota GPU harian terbatas untuk akun gratis — kalau habis, request generation akan
  gagal/ditolak sampai kuota reset (~24 jam). Kalau butuh lebih banyak, pertimbangkan
  HF PRO ($9/bulan, 8x kuota) atau turunkan `duration` di decorator @spaces.GPU.

Struktur folder Space yang dibutuhkan:
    app.py                  <- file ini
    requirements.txt
    data/
        chunks.json         <- hasil dari notebook (Step 4b)
        faiss_index.bin     <- hasil dari notebook (Step 6)
"""

import spaces  # HARUS import paling pertama, sebelum torch/transformers/faiss/dll,
                # supaya CUDA belum ke-init duluan sebelum spaces masang patch-nya

import json
import os

import faiss
import gradio as gr
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
ARTIFACT_DIR = "./data"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"

# Dengan ZeroGPU (VRAM besar), model 7B nyaman dipakai. Kalau mau lebih hemat kuota
# (generation lebih cepat = durasi @spaces.GPU lebih pendek = kuota harian lebih awet),
# turunkan ke 'Qwen/Qwen2.5-3B-Instruct'.
GEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

MAX_CHARS_PER_CHUNK = 600
TOP_K = 5
GPU_DURATION = 60  # detik maksimal per pemanggilan @spaces.GPU; naikkan kalau sering timeout

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
# Load artifacts & model (sekali saja saat Space start)
# ---------------------------------------------------------------------------
print("Memuat chunks...")
with open(os.path.join(ARTIFACT_DIR, "chunks.json"), "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

print("Memuat FAISS index...")
index = faiss.read_index(os.path.join(ARTIFACT_DIR, "faiss_index.bin"))

print("Memuat model embedding (CPU, gak butuh ZeroGPU)...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

print(f"Memuat model generation ({GEN_MODEL_NAME})...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},  # eksplisit ke GPU index 0, lebih kompatibel dgn ZeroGPU dibanding 'auto'
)
print("Model siap.")


# ---------------------------------------------------------------------------
# Retrieval (jalan di CPU, gak perlu ZeroGPU — cepat untuk skala chunk kita)
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
# Generation — INI yang butuh GPU, jadi didekorasi @spaces.GPU
# ---------------------------------------------------------------------------
@spaces.GPU(duration=GPU_DURATION)
def generate_answer(query, retrieved_chunks, max_new_tokens=800):
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
    prompt = gen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)

    with torch.no_grad():
        output_ids = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=gen_tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    answer = gen_tokenizer.decode(generated, skip_special_tokens=True).strip()

    del inputs, output_ids, generated
    torch.cuda.empty_cache()

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
