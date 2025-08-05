from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

def msg_to_toks(messages, tokenizer, device="cuda"):
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512)
    return inputs.input_ids.to(device)

def get_logprobs(model, tokenizer, query_messages, response):
    device = next(model.parameters()).device  # Get model's device
    model.eval()
    full_messages = query_messages + [{"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    query_text = tokenizer.apply_chat_template(query_messages, tokenize=False, add_generation_prompt=True)
    full_inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    query_inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=512)
    full_ids = full_inputs.input_ids.to(device)
    query_len = query_inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits
        response_logits = logits[0, query_len-1:-1]
        response_targets = full_ids[0, query_len:]
        log_probs = F.log_softmax(response_logits, dim=-1)
        target_log_probs = log_probs.gather(1, response_targets.unsqueeze(1)).squeeze(1)
        return target_log_probs.sum().item()

def compute_gradients(model, tokenizer, messages):
    device = next(model.parameters()).device  # Get model's device
    model.zero_grad()
    model.train()
    input_ids = msg_to_toks(messages, tokenizer, device)
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()

def update_model_weights(model, lr):
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.data += lr * param.grad

def initialize_model():
    global model, tokenizer
    MODEL_NAME = "gpt2"
    # MODEL_NAME = "google/gemma-3-4b-it"
    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Model loaded successfully!")

@app.route('/compute_logprobs', methods=['POST'])
def compute_logprobs():
    try:
        data = request.json
        
        # Extract the inputs
        train_q = data.get('train_q', '').strip()
        train_a = data.get('train_a', '').strip()
        test_q = data.get('test_q', '').strip()
        test_a = data.get('test_a', '').strip()
        
        # Check if any input is empty
        if not all([train_q, train_a, test_q, test_a]):
            return jsonify({
                'error': 'All fields must be filled',
                'before_logprob': None,
                'after_logprob': None
            })
        
        # Prepare messages in the expected format
        train_messages = [
            {"role": "user", "content": train_q},
            {"role": "assistant", "content": train_a}
        ]
        test_query = [{"role": "user", "content": test_q}]
        
        # Compute before logprobs
        logger.info("Computing before logprobs...")
        before_logprob = get_logprobs(model, tokenizer, test_query, test_a)
        
        # Update model with training data
        logger.info("Computing gradients and updating model...")
        compute_gradients(model, tokenizer, train_messages)
        update_model_weights(model, 1e-3)  # LR = 1e-3
        
        # Compute after logprobs with updated model
        logger.info("Computing after logprobs...")
        after_logprob = get_logprobs(model, tokenizer, test_query, test_a)
        
        logger.info(f"Before: {before_logprob:.4f}, After: {after_logprob:.4f}")
        
        return jsonify({
            'before_logprob': round(before_logprob, 4),
            'after_logprob': round(after_logprob, 4),
            'difference': round(after_logprob - before_logprob, 4)
        })
        
    except Exception as e:
        logger.error(f"Error computing logprobs: {str(e)}")
        return jsonify({
            'error': str(e),
            'before_logprob': None,
            'after_logprob': None
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    initialize_model()
    app.run(debug=True, host='0.0.0.0', port=5000)