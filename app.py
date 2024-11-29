import os
import time
import requests
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS

# Create Flask app instance
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  # Enable CORS for all routes

# Hugging Face API settings
HF_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Ensure the 'static' folder exists to store images
if not os.path.exists('static'):
    os.makedirs('static')

def generate_image(prompt, output_path="static/generated_image.png"):
    """Generate an image using Hugging Face Inference API with retry logic."""
    retries = 5  # Number of retries
    delay = 60  # Delay in seconds between retries (you can adjust this)
    
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})
            
            # Check if the model is ready
            if response.status_code == 503:
                # Check if model is loading, wait and retry
                error_info = response.json()
                print(f"Attempt {attempt + 1}: Model is loading, estimated time: {error_info['error'].get('estimated_time', 'N/A')}")
                if attempt < retries - 1:
                    time.sleep(delay)  # Wait before retrying
                    continue
                else:
                    raise Exception("Model failed to load after several retries.")
            
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return output_path
            else:
                raise Exception(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                raise Exception(f"Failed to generate image after {retries} attempts: {str(e)}")

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Handle image generation requests."""
    data = request.json
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    try:
        image_path = generate_image(prompt)
        return jsonify({"image_url": f"/static/{os.path.basename(image_path)}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5500, debug=True)
