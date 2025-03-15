# make sure you've installed gradio : "pip install gradio"
import gradio as gr
import requests
from bs4 import BeautifulSoup
import random

def classify_website(url, deepfake, nsfw, violence, hate_speech):
    # Fetch the website content
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        content = response.text  # Get raw HTML content
    except Exception as e:
        return f"Error fetching website: {e}", ""
    
    # Mock classification scores (replace with your model's inference)
    results = {}
    if deepfake:
        results['Deepfake'] = random.uniform(0, 1)
    if nsfw:
        results['PG/18+ Content'] = random.uniform(0, 1)
    if violence:
        results['Violence/Self-Harm'] = random.uniform(0, 1)
    if hate_speech:
        results['Hate Speech'] = random.uniform(0, 1)
    
    report = "\n".join([f"{key}: {value:.2f}" for key, value in results.items()])
    
    return report, content  # Return full website content

def load_content(url):
    if url.lower().endswith(".jpg"):  # Check if the URL points to a .jpg image
        return f'<img src="{url}" alt="Image Preview" style="max-width: 100%;">'
    else:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img_np = np.array(img)

            return response.text, img_np  # Load full website HTML
        except Exception as e:
            return f"Error loading content: {e}"

with gr.Blocks(css=".gradio-container {font-family: Arial;}") as demo:
    gr.Markdown("## 🌐 AI-Powered Content Safety Browser")
    
    with gr.Row():
        with gr.Column(scale=2):  # Left panel
            url_input = gr.Textbox(label="Enter Website or Image URL", placeholder="https://example.com or https://example.com/image.jpg")
            go_button = gr.Button("Go")
            deepfake = gr.Checkbox(label="Detect Deepfakes", value=True)
            nsfw = gr.Checkbox(label="Detect PG/18+ Content", value=True)
            violence = gr.Checkbox(label="Detect Violent Scenes", value=True)
            hate_speech = gr.Checkbox(label="Detect Hate Speech", value=True)
            classify_button = gr.Button("Analyze Website")
            report_output = gr.Textbox(label="Classification Report", interactive=False)
        
        with gr.Column(scale=8):  # Right panel
            website_preview = gr.HTML(label="Website/Image Preview")
    
    classify_button.click(classify_website, [url_input, deepfake, nsfw, violence, hate_speech], [report_output, website_preview])
    go_button.click(load_content, url_input, website_preview)

demo.launch()
