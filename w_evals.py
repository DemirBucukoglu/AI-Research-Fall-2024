import os
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoModelForVision2Seq
)
import torch
from PIL import Image

# Define model configurations
models_config = {
    "1": {
        "name": "GIT Large COCO",
        "processor": lambda: AutoProcessor.from_pretrained("microsoft/git-large-coco"),
        "model": lambda: AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")
    },
    "2": {
        "name": "ViT-GPT2",
        "processor": lambda: ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning"),
        "model": lambda: VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning"),
        "tokenizer": lambda: AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    },
    "3": {
        "name": "BLIP",
        "processor": lambda: BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
        "model": lambda: BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    },
    "4": {
        "name": "Kosmos-2",
        "processor": lambda: AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224"),
        "model": lambda: AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
    }
}

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor
def load_model(model_key):
    config = models_config[model_key]
    processor = config["processor"]()
    model = config["model"]().to(device)
    tokenizer = config.get("tokenizer", lambda: None)()
    return model, processor, tokenizer, config["name"]

# Generate captions for an image
def generate_caption(image_path, model, processor, tokenizer, model_key):
    try:
        image = Image.open(image_path).convert("RGB")
        if model_key == "1":  # GIT Large COCO
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=300, num_beams=4, num_return_sequences=4)
            captions = processor.batch_decode(outputs, skip_special_tokens=True)
            captions = [caption for caption in captions if "google street view" not in caption.lower()]
            return captions[0] if captions else "No meaningful description generated."
        elif model_key == "2":  # ViT-GPT2
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            output_ids = model.generate(pixel_values, max_length=300, num_beams=10)
            return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        elif model_key == "3":  # BLIP
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_length=300, num_beams=10)
            return processor.decode(outputs[0], skip_special_tokens=True).strip()
        elif model_key == "4":  # Kosmos-2
            prompt = "Describe the surroundings, including buildings, streets, people, and nature."
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            generated_ids = model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                max_new_tokens=150, num_beams=10
            )
            return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return "No meaningful description generated."

# Compute BLEU score
def compute_bleu_score(generated_caption, reference_captions):
    smoothing_function = SmoothingFunction().method7
    return sentence_bleu(
        [ref.split() for ref in reference_captions],
        generated_caption.split(),
        smoothing_function=smoothing_function
    )

# Compute ROUGE scores
def compute_rouge_score(generated_caption, reference_captions):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(generated_caption, ref) for ref in reference_captions]
    return {
        metric: sum(score[metric].fmeasure for score in scores) / len(scores)
        for metric in ['rouge1', 'rouge2', 'rougeL']
    }

# Evaluate models
def evaluate_models(folder_path, ground_truths, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Model Evaluation Results:\n\n")

        for model_key, config in models_config.items():
            print(f"\nLoading model: {config['name']}...")
            model, processor, tokenizer, model_name = load_model(model_key)
            print(f"Model {model_name} loaded successfully.\n")

            f.write(f"Evaluating Model: {model_name}\n")
            f.write("=" * 60 + "\n")

            bleu_scores = []
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            image_count = 0

            for image_file in os.listdir(folder_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_count += 1
                    print(f"Processing image {image_count}: {image_file}...")
                    image_path = os.path.join(folder_path, image_file)
                    generated_caption = generate_caption(image_path, model, processor, tokenizer, model_key)

                    # Skip evaluation if the caption is invalid
                    if generated_caption == "No meaningful description generated.":
                        print(f"Skipped evaluation for image {image_file} due to invalid caption.")
                        continue

                    key = os.path.splitext(image_file)[0]
                    if key in ground_truths:
                        references = ground_truths[key]
                        bleu = compute_bleu_score(generated_caption, references)
                        rouge = compute_rouge_score(generated_caption, references)

                        bleu_scores.append(bleu)
                        for metric in rouge:
                            rouge_scores[metric].append(rouge[metric])

                        print(f"Completed evaluation for image {image_file}: BLEU = {bleu:.4f}")

                        f.write(f"Image: {image_file}\n")
                        f.write(f"Generated Caption: {generated_caption}\n")
                        f.write(f"BLEU: {bleu:.4f}\n")
                        f.write(f"ROUGE-1: {rouge['rouge1']:.4f}\n")
                        f.write(f"ROUGE-2: {rouge['rouge2']:.4f}\n")
                        f.write(f"ROUGE-L: {rouge['rougeL']:.4f}\n")
                        f.write("-" * 60 + "\n")

            # Overall scores
            f.write(f"\nOverall Scores for {model_name}:\n")
            if bleu_scores:
                avg_bleu = sum(bleu_scores) / len(bleu_scores)
                f.write(f"Average BLEU: {avg_bleu:.4f}\n")
                print(f"Average BLEU for {model_name}: {avg_bleu:.4f}")
            else:
                f.write("No valid captions for BLEU evaluation.\n")

            for metric in rouge_scores:
                if rouge_scores[metric]:
                    avg_rouge = sum(rouge_scores[metric]) / len(rouge_scores[metric])
                    f.write(f"Average {metric.upper()}: {avg_rouge:.4f}\n")
                    print(f"Average {metric.upper()} for {model_name}: {avg_rouge:.4f}")
                else:
                    f.write(f"No valid captions for {metric.upper()} evaluation.\n")
            f.write("=" * 60 + "\n\n")

        print("\nAll evaluations completed. Results saved to:", output_path)

# Main execution
folder_path = r"C:\Users\Demir\Desktop\CI and CPSC\CI FALL 2024\1 image"
ground_truths_path = r"C:\Users\Demir\Desktop\CI and CPSC\CI FALL 2024\ci final\ground_truths.json"
output_path = r"C:\code\CI\model_evaluation_results.txt"

with open(ground_truths_path, "r", encoding="utf-8") as f:
    ground_truths = json.load(f)

evaluate_models(folder_path, ground_truths, output_path)
