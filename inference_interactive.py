import torch
from transformers import AutoTokenizer
from model import load_model
import argparse
import readline  # enables arrow-key history on Unix
from termcolor import colored

def load_pipeline(model_path, model_name="roberta-base", max_length=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model(model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return tokenizer, model, device

def predict(text, tokenizer, model, device, max_length=512):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=1).squeeze()
        label = torch.argmax(probs).item()
        confidence = probs[label].item()

    label_str = "AI-Generated" if label == 1 else "Human-Written"
    color = "red" if label == 1 else "green"

    print("\n==============================")
    print(f"📝  {colored('Input:', attrs=['bold'])} {text[:120]}{'...' if len(text) > 120 else ''}")
    print(f"🔍  {colored('Prediction:', 'cyan')} {colored(label_str, color, attrs=['bold'])}")
    print(f"📊  {colored('Confidence:', 'cyan')} {confidence:.4f}")
    print("==============================\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/roberta_ai_detector.pt")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    tokenizer, model, device = load_pipeline(
        model_path=args.model_path,
        model_name=args.model_name,
        max_length=args.max_length
    )

    print(colored("\n🧠 AI Detector is running. Type or paste your text below.", "yellow"))
    print(colored("🔁 Type 'exit' or press Ctrl+C to quit.\n", "yellow", attrs=["dark"]))

    try:
        while True:
            user_input = input(colored("💬 > ", "blue"))
            if user_input.lower().strip() in ["exit", "quit"]:
                print(colored("👋 Goodbye!", "yellow"))
                break
            elif len(user_input.strip()) < 10:
                print(colored("⚠️  Please enter at least a few words.\n", "magenta"))
                continue
            predict(user_input.strip(), tokenizer, model, device, max_length=args.max_length)
    except KeyboardInterrupt:
        print(colored("\n👋 Interrupted. Exiting...\n", "yellow"))

if __name__ == "__main__":
    main()
