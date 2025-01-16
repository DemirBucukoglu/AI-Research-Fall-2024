import os
from PIL import Image
import torch
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



# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configurations
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


active_model = None
active_processor = None
active_tokenizer = None


def load_model(model_key):
    global active_model, active_processor, active_tokenizer
    config = models_config.get(model_key)
    if not config:
        raise ValueError(f"Model '{model_key}' not configured.")
    active_processor = config["processor"]()
    active_model = config["model"]().to(device)
    if "tokenizer" in config:
        active_tokenizer = config["tokenizer"]()
    print(f"Loaded model: {config['name']}")  


# Function to generate the captions
def generate_caption(image_path, model_key):
    image = Image.open(image_path).convert("RGB")
    if model_key == "1":  # GIT Large COCO
        inputs = active_processor(images=image, return_tensors="pt").to(device)
        output = active_model.generate(
            **inputs,
            max_new_tokens=150,  
            num_beams=10,       
            temperature=0.6,    
            top_p=0.9,          
            repetition_penalty=1.2  
        )
        return active_processor.batch_decode(output, skip_special_tokens=True)[0]
    
    elif model_key == "2":  # ViT-GPT2
        pixel_values = active_processor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = active_model.generate(
            pixel_values,
            max_length=150,     
            num_beams=10,       
            temperature=0.7,    
            repetition_penalty=1.2  
        )
        return active_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    
    elif model_key == "3":  # BLIP
        inputs = active_processor(
            images=image,
            return_tensors="pt"
        ).to(device)
        output = active_model.generate(
            **inputs,
            max_length=150,     
            num_beams=10,       
            temperature=0.6,    
            repetition_penalty=1.2  
        )
        return active_processor.decode(output[0], skip_special_tokens=True).strip()
    
    elif model_key == "4":  # Kosmos-2
        prompt = "<grounding>An image showing a detailed and accurate description of the scene"
        inputs = active_processor(text=prompt, images=image, return_tensors="pt").to(device)
        generated_ids = active_model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            max_new_tokens=150,  
            num_beams=10,         
            temperature=0.6,     
            top_p=0.9,            
            repetition_penalty=1.2  
            repetition_penalty = 1.2
        )
        return active_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]



def process_images(folder_path, model_key, output_file=None):
    config = models_config.get(model_key)
    if output_file is None:
        output_file = os.path.join(folder_path, f"{config['name'].replace(' ', '_')}_output.txt")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\n===== Model: {config['name']} =====\n")
    load_model(model_key)

    image_files = [
        f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
    ]
    with open(output_file, "a", encoding="utf-8") as f:
        for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                caption = generate_caption(image_path, model_key)
                output = f"{image_file} ({models_config[model_key]['name']}): {caption}\n"
                print(output)
                f.write(output)

# Run all models sequentially
def run_all_models(folder_path):
    output_file = os.path.join(r"PATH", "PATH")
    if os.path.exists(output_file):
        os.remove(output_file)  # removes the old output file
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n")
    for model_key in models_config.keys():
        process_images(folder_path, model_key, output_file) # processes images one by one from the file
    print(f"All models have been processed. Results saved to {output_file}")


# input folder where the car images are 
folder_path = r"PATH"

# shows the options
print("Choose a model:")
for key, config in models_config.items():
    print(f"{key}: {config['name']}")
print("A: Run all models sequentially")

# Prompt user to select a model
model_key = input("Enter the number corresponding to the model you want to use or 'A' for all: ").strip()
if model_key == "A":
    run_all_models(folder_path)
elif model_key in models_config:
    process_images(folder_path, model_key)
else:
    print("Invalid selection. Exiting.")
