import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to your fine-tuned checkpoint
#MODEL_PATH = "./output/tamusa_sft_model/checkpoint-800"
MODEL_PATH = "./output/tamusa_final_model"


# If you later create a final merged folder, replace with:
# MODEL_PATH = "./output/tamusa_final_model"


def load_model_and_tokenizer():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Ensure padding token is set (important for GPT-style models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32  # Safe for CPU
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    return model, tokenizer, device


def generate_response(model, tokenizer, device, user_input):
    # Match your training format exactly
    formatted_prompt = f"Instruction: {user_input}\nResponse:"

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from output
    response = decoded.replace(formatted_prompt, "").strip()

    return response


def main():
    model, tokenizer, device = load_model_and_tokenizer()

    print("\nTAMUSA Chat Model Ready!")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = generate_response(model, tokenizer, device, user_input)

        print("\nTAMUSA Bot:", response)
        print("-" * 60)


if __name__ == "__main__":
    main()
