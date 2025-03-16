# Make sure you've installed gradio: "pip install gradio"
import gradio as gr
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import ResNetForImageClassification

# Load Pretrained Model
model = ResNetForImageClassification.from_pretrained("./outputs/checkpoint-1750", num_labels=2)

# Define label names (modify based on your dataset)
LABELS = ["Safe", "Harmful"]

# Function to classify an image (URL or upload)
def predict(img_np: np.ndarray):
    try:
        # Convert to tensor and preprocess
        img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float()

        # Run inference
        with torch.no_grad():
            logits = model(img_tensor).logits

        # Get predictions
        probs = torch.sigmoid(logits).numpy()[0]
        predictions = {LABELS[i]: round(probs[i], 2) for i in range(len(LABELS))}
        
        return "\n".join([f"{label}: {score}" for label, score in predictions.items()]), img_np  # Return results + image preview
    except Exception as e:
        return f"Error processing image: {e}", None

# Function to fetch an image from a URL and classify it
def load_content(url):
    if url.lower().endswith((".jpg", ".png", ".jpeg")):  # If direct image URL
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")

            # Convert to NumPy array
            img_np = np.array(img)

            return img_np, predict(img_np)[0]  # Preview image + classification result
        except Exception as e:
            return None, f"Error loading image: {e}"
    else:
        return None, "Invalid URL (must be a direct image link)"

# Gradio UI
with gr.Blocks(css=".gradio-container {font-family: Arial;}") as demo:
    gr.Markdown("## 🌐 AI-Powered Content Safety Browser")

    with gr.Row():
        with gr.Column(scale=2):  # Left panel
            url_input = gr.Textbox(label="Enter Image URL", placeholder="https://example.com/image.jpg")
            go_button = gr.Button("Get Image")
            image_input = gr.Image(label="Upload Image", type="numpy")  # Auto-triggers analysis
            report_output = gr.Textbox(label="Classification Report", interactive=False)
        
        with gr.Column(scale=8):  # Right panel
            website_preview = gr.Image(label="Image Preview")  # Displays preview image

    # When "Get Image" is clicked, fetch + classify image from URL
    go_button.click(load_content, url_input, [website_preview, report_output])

    # When an image is uploaded, classify it automatically
    image_input.change(predict, image_input, [report_output, website_preview])

demo.launch()
