from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from flasgger import Swagger
from flask_cors import CORS
import torch
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests
swagger = Swagger(app)

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>GPT-2 Text Generator</title>
        <style>
            body {
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                background: #f5f7fa;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                color: #333;
            }
            .container {
                background: #fff;
                padding: 30px 40px;
                border-radius: 8px;
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                max-width: 600px;
                width: 90%;
            }
            h1 {
                text-align: center;
                color: #2c3e50;
                margin-bottom: 24px;
                font-weight: 700;
            }
            label {
                font-weight: 600;
                display: block;
                margin-bottom: 8px;
                margin-top: 20px;
                color: #34495e;
            }
            textarea, input[type=number] {
                width: 100%;
                padding: 12px 15px;
                font-size: 16px;
                border: 1.5px solid #ddd;
                border-radius: 6px;
                box-sizing: border-box;
                resize: vertical;
                transition: border-color 0.3s ease;
            }
            textarea:focus, input[type=number]:focus {
                border-color: #3498db;
                outline: none;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 14px 0;
                width: 100%;
                font-size: 18px;
                border-radius: 6px;
                cursor: pointer;
                margin-top: 30px;
                font-weight: 700;
                transition: background-color 0.3s ease;
            }
            button:hover {
                background-color: #2980b9;
            }
            h2 {
                margin-top: 40px;
                font-weight: 700;
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 6px;
            }
            pre {
                background: #ecf0f1;
                padding: 20px;
                border-radius: 6px;
                white-space: pre-wrap;
                word-wrap: break-word;
                max-height: 300px;
                overflow-y: auto;
                font-size: 15px;
                color: #2c3e50;
                margin-top: 10px;
            }
            .error {
                color: #e74c3c;
                font-weight: 700;
                margin-top: 15px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>GPT-2 Text Generation</h1>
            <form id="generate-form">
                <label for="prompt">Enter your prompt:</label>
                <textarea id="prompt" name="prompt" rows="5" placeholder="Type your prompt here..." required></textarea>
                
                <label for="max_length">Max tokens to generate (10-100):</label>
                <input type="number" id="max_length" name="max_length" value="50" min="10" max="100" />
                
                <button type="submit">Generate Text</button>
            </form>
            
            <h2>Generated Text:</h2>
            <pre id="result"></pre>
            <div id="error-msg" class="error"></div>
        </div>

        <script>
            const form = document.getElementById('generate-form');
            const resultBox = document.getElementById('result');
            const errorBox = document.getElementById('error-msg');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                resultBox.textContent = '';
                errorBox.textContent = '';

                const prompt = document.getElementById('prompt').value.trim();
                const max_length = document.getElementById('max_length').value;

                if (!prompt) {
                    errorBox.textContent = "Prompt cannot be empty.";
                    return;
                }

                const formData = new FormData();
                formData.append('prompt', prompt);
                formData.append('max_length', max_length);

                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    if (response.ok) {
                        resultBox.textContent = data.generated_text;
                    } else {
                        errorBox.textContent = data.error || 'An error occurred.';
                    }
                } catch (error) {
                    errorBox.textContent = 'Error: ' + error.message;
                }
            });
        </script>
    </body>
    </html>
    '''


@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate text using GPT-2
    ---
    parameters:
      - name: prompt
        in: formData
        type: string
        required: true
        description: Text prompt to generate from
      - name: max_length
        in: formData
        type: integer
        required: false
        default: 50
        description: Maximum number of tokens to generate (10-100)
    responses:
      200:
        description: Generated text
        schema:
          type: object
          properties:
            generated_text:
              type: string
              example: Once upon a time...
      400:
        description: Invalid input
        schema:
          type: object
          properties:
            error:
              type: string
              example: Prompt cannot be empty.
      500:
        description: Internal server error
        schema:
          type: object
          properties:
            error:
              type: string
              example: Failed to generate text.
    """
    data = request.form

    prompt = data.get("prompt", "").strip()
    max_length = data.get("max_length", 50)

    # Input validation
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    try:
        max_length = int(max_length)
        if max_length < 10 or max_length > 100:
            return jsonify({"error": "max_length must be between 10 and 100."}), 400
    except ValueError:
        return jsonify({"error": "max_length must be an integer."}), 400

    logging.info(f"Received prompt: '{prompt}', max_length: {max_length}")

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=max_length)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logging.info("Text generation successful")
        return jsonify({"generated_text": text})

    except Exception as e:
        logging.error(f"Error during generation: {e}")
        return jsonify({"error": "Failed to generate text."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)