import torch
import faiss
import pickle
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ==========================================================
# CONFIG
# ==========================================================

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_PATH = "./output/tamusa_v3_final"

INDEX_FILE = "tamusa_index.faiss"
META_FILE = "tamusa_chunks.pkl"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


# ==========================================================
# DEVICE DETECTION
# ==========================================================

def get_device():
    if torch.cuda.is_available():
        print("GPU detected:", torch.cuda.get_device_name(0))
        return "cuda"
    else:
        print("No GPU detected. Running in CPU mode.")
        return "cpu"


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model_and_tokenizer():

    device = get_device()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ================= GPU MODE =================
    if device == "cuda":
        print("Loading base model in 4-bit (GPU mode)...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

    # ================= CPU MODE =================
    else:
        print("Loading base model in full precision (CPU mode)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        base_model.to("cpu")

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)

    model.eval()
    print("Model ready!")

    return model, tokenizer


# ==========================================================
# LOAD RETRIEVAL SYSTEM
# ==========================================================

def load_retrieval_system():
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_FILE)

    print("Loading metadata...")
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)

    documents = metadata["documents"]
    sources = metadata["sources"]

    print("Retrieval system ready!")
    return embed_model, index, documents, sources


def retrieve_context(embed_model, index, documents, sources, query, k=TOP_K):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    retrieved_chunks = []
    retrieved_sources = []

    for idx in indices[0]:
        retrieved_chunks.append(documents[idx])
        retrieved_sources.append(sources[idx])

    return retrieved_chunks, retrieved_sources


# ==========================================================
# GENERATE
# ==========================================================

def generate_response(model, tokenizer,
                      embed_model, index, documents, sources,
                      user_input):

    chunks, chunk_sources = retrieve_context(
        embed_model, index, documents, sources, user_input
    )

    context_text = "\n\n".join(chunks)

    prompt = f"""### System:
You are the official TAMUSA assistant for Texas A&M University–San Antonio.

You MUST answer ONLY using the provided Official Context below.
If the answer is not explicitly contained in the context, say:
"I do not have verified information from official TAMUSA sources."

Never guess.
Never mention other campuses.
Never fabricate catalog information.

### Official Context:
{context_text}

### User:
{user_input}

### Assistant:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.0,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded[len(prompt):].strip()

    for stop_token in ["### System:", "### User:"]:
        if stop_token in response:
            response = response.split(stop_token)[0].strip()

    unique_sources = list(set(chunk_sources))

    citation_block = "\n\nSources:\n"
    for src in unique_sources:
        citation_block += f"- {src}\n"

    return response + citation_block


# ==========================================================
# CHAT LOOP
# ==========================================================

def main():

    model, tokenizer = load_model_and_tokenizer()
    embed_model, index, documents, sources = load_retrieval_system()

    print("\nTAMUSA RAG System Ready!")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        response = generate_response(
            model, tokenizer,
            embed_model, index, documents, sources,
            user_input
        )

        print("\nTAMUSA Bot:\n")
        print(response)
        print("-" * 80)


if __name__ == "__main__":
    main()
