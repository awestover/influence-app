import random
from math import log
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import logging
import argparse
LRS = [1e-5, 1e-4, 1e-3]
assert LRS[0] < LRS[1] < LRS[2]
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')
MOCK = False
DO_SAMPLE = False
LORA = False
DEFAULT_MODEL_NAME = "google/gemma-3-12b-it"

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
        # GET RID OF EOS
        response_logits = logits[0, query_len-1:-1]
        response_targets = full_ids[0, query_len:-3]
        import ipdb; ipdb.set_trace()
        log_probs = F.log_softmax(response_logits, dim=-1)
        target_log_probs = log_probs.gather(1, response_targets.unsqueeze(1)).squeeze(1)
        return target_log_probs.sum().item()

def get_yes_no_logprobs(model, tokenizer, query_messages, _):
    """Get log probabilities for yes/Yes/no/No tokens with proper normalization"""
    device = next(model.parameters()).device
    model.eval()
    
    # Get the token IDs for yes/no variants - order matters for the computation
    yes_tokens = ["yes", "Yes", "no", "No"]
    token_mapping = {}
    token_ids = []
    
    for token in yes_tokens:
        # Get token ID, handling potential multiple tokens
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) == 1:  # Only use single-token words
            token_mapping[token] = encoded[0]
            token_ids.append(encoded[0])
    
    # We need all four tokens for proper computation
    required_tokens = ["yes", "Yes", "no", "No"]
    missing_tokens = [t for t in required_tokens if t not in token_mapping]
    if missing_tokens:
        return {"error": f"Could not find token IDs for: {missing_tokens}"}
    
    # Format query for generation
    query_text = tokenizer.apply_chat_template(query_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1]  # Get logits for next token prediction
        
        # Get individual logits
        yes_logit = logits[token_mapping["yes"]]
        Yes_logit = logits[token_mapping["Yes"]]
        no_logit = logits[token_mapping["no"]]
        No_logit = logits[token_mapping["No"]]
        
        # Compute Z = exp(logit(no)) + exp(logit(yes)) + exp(logit(Yes)) + exp(logit(No))
        Z = torch.exp(yes_logit) + torch.exp(Yes_logit) + torch.exp(no_logit) + torch.exp(No_logit)
        
        # Compute log probability of "yes" variants: log((exp(logit(yes)) + exp(logit(Yes))) / Z)
        yes_numerator = torch.exp(yes_logit) + torch.exp(Yes_logit)
        combined_yes_logprob = torch.log(yes_numerator / Z)
        
        # Compute log probability of "no" variants: log((exp(logit(no)) + exp(logit(No))) / Z)
        no_numerator = torch.exp(no_logit) + torch.exp(No_logit)
        combined_no_logprob = torch.log(no_numerator / Z)
        
        return {
            "yes_logprob": combined_yes_logprob.item(),
            "no_logprob": combined_no_logprob.item()
        }
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
        for _, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.data -= lr * param.grad
def generate_text(model, tokenizer, query_messages, _, max_new_tokens=20, temperature=0.7):
    """Generate text from the model given query messages"""
    device = next(model.parameters()).device
    model.eval()
    # Format the query with generation prompt
    query_text = tokenizer.apply_chat_template(query_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        # Extract only the newly generated tokens
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text
def initialize_model(MODEL_NAME=DEFAULT_MODEL_NAME):
    global model, tokenizer
    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    
    if LORA:
        logger.info("Initializing model with LORA adapter")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # rank
            lora_alpha=32,  # scaling parameter
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(base_model, lora_config)
        logger.info("LORA adapter added successfully!")
    else:
        logger.info("Using full model (no LORA)")
        model = base_model
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Model loaded successfully!")
    return model, tokenizer

def backend(fn):
    logger.info(f"Running {fn.__name__}")
    data = request.json
    parsed = {x: data.get(x, '').strip() for x in ["train_q", "train_a", "test_q", "test_a"]}
    if not all(parsed.values()):
        return jsonify({ 'error': 'Missing fields', 'results': None })
    train_messages = [
        {"role": "user", "content": parsed["train_q"]},
        {"role": "assistant", "content": parsed["train_a"]}
    ]
    test_query = [{"role": "user", "content": parsed["test_q"]}]
    test_a = parsed["test_a"]
    try:
        results = {"0": fn(model, tokenizer, test_query, test_a)}
        compute_gradients(model, tokenizer, train_messages)
        for lri, lr in enumerate(LRS):
            logger.info(f"Running with {lr}")
            lrdiff = lr if lri == 0 else LRS[lri] - LRS[lri-1]
            update_model_weights(model, lrdiff)
            results[str(lr)] = fn(model, tokenizer, test_query, test_a)
        update_model_weights(model, -LRS[-1])
        return jsonify(results)        
    except Exception as e:
        return jsonify({'error': str(e), 'results': None }), 500
@app.route('/compute_logprobs', methods=['POST'])
def compute_logprobs():
    if MOCK:
        return jsonify({"0": -random.random(), "1e-5": -random.random(), "1e-4": -random.random(), "1e-3": -random.random()})
    return backend(get_logprobs)
@app.route('/generate_completions', methods=['POST'])
def generate_completions():
    if MOCK:
        return jsonify({"0": "Hello, world!", "1e-5": "Hello, world!", "1e-4": "Hello, world!", "1e-3": "Hello, world!"})
    return backend(generate_text)
@app.route('/compute_yes_no_probs', methods=['POST'])
def compute_yes_no_probs():
    if MOCK:
        dummy = {}
        for lr in LRS:
            yespr = random.random()
            dummy[str(lr)] = {"yes_logprob": log(yespr), "no_logprob": log(1-yespr)}
        return jsonify(dummy)
    return backend(get_yes_no_logprobs)
@app.route('/reset_model', methods=['POST'])
def reset_model():
    """Reset/reinitialize the model to its original state"""
    if not MOCK:
        initialize_model() 
    return jsonify({"success": True, "message": "Model reset successfully"})

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the influence app with optional mock mode')
    parser.add_argument('--mock', action='store_true', help='Enable mock mode for testing')
    parser.add_argument('--sample', action='store_true', help='Enable sampling during text generation')
    parser.add_argument('--lora', action='store_true', help='Enable LORA adapter')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_NAME, help='Model name to load')
    args = parser.parse_args()
    MOCK = args.mock
    DO_SAMPLE = args.sample
    LORA = args.lora
    DEFAULT_MODEL_NAME = args.model
    logger.info("Starting Flask app...")
    if not MOCK:
        initialize_model()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
