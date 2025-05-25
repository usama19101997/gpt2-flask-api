from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)

# Model and tokenizer setup
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Flask app
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        prompt = data["prompt"]
        max_length = int(data.get("max_length", 50))
        temperature = float(data.get("temperature", 1.0))
        top_k = int(data.get("top_k", 50))
        num_return_sequences = int(data.get("num_return_sequences", 1))

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=num_return_sequences
        )

        texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        logging.info(f"Generated {num_return_sequences} result(s) for prompt: {prompt}")

        return jsonify({"generated_texts": texts})

    except Exception as e:
        logging.error("Error during generation: " + str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
