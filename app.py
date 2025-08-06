from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import copy

"""
Model influence computation using deep copying approach for clean state isolation.
"""
LRS = [1e-5, 1e-4, 1e-3]
assert LRS[0] < LRS[1] < LRS[2]
MODEL_NAME = "google/gemma-3-4b-it"
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
base_model = None
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

def get_yes_no_logprobs(model, tokenizer, query_messages):
    """Get log probabilities for yes/Yes/no/No tokens only"""
    device = next(model.parameters()).device
    model.eval()
    
    # Get the token IDs for yes/no variants
    yes_tokens = ["yes", "Yes", "no", "No"]
    token_ids = []
    for token in yes_tokens:
        # Get token ID, handling potential multiple tokens
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) == 1:  # Only use single-token words
            token_ids.append(encoded[0])
    
    if len(token_ids) < 2:  # Need at least 2 tokens to be meaningful
        return {"error": "Could not find sufficient yes/no token IDs"}
    
    # Format query for generation
    query_text = tokenizer.apply_chat_template(query_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1]  # Get logits for next token prediction
        
        # Extract logits for yes/no tokens only
        yes_no_logits = logits[token_ids]
        
        # Apply softmax only to these tokens
        yes_no_probs = F.softmax(yes_no_logits, dim=-1)
        yes_no_logprobs = F.log_softmax(yes_no_logits, dim=-1)
        
        # Create result dictionary
        result = {}
        for i, token_id in enumerate(token_ids):
            token_text = tokenizer.decode([token_id])
            result[token_text] = {
                'token_id': token_id,
                'logprob': yes_no_logprobs[i].item(),
                'prob': yes_no_probs[i].item()
            }
        
        return result
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
                param.data -= lr * param.grad
def generate_text(model, tokenizer, query_messages, max_new_tokens=50, temperature=0.7):
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
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        # Extract only the newly generated tokens
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text
def initialize_model():
    global base_model, tokenizer
    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Model loaded successfully!")

@app.route('/compute_logprobs', methods=['POST'])
def compute_logprobs():
    try:
        data = request.json
        train_q = data.get('train_q', '').strip()
        train_a = data.get('train_a', '').strip()
        test_q = data.get('test_q', '').strip()
        test_a = data.get('test_a', '').strip()
        if not all([train_q, train_a, test_q, test_a]):
            return jsonify({
                'error': 'All fields must be filled',
                'results': None
            })
        
        train_messages = [
            {"role": "user", "content": train_q},
            {"role": "assistant", "content": train_a}
        ]
        test_query = [{"role": "user", "content": test_q}]
        
        results = {}
        # Create a deep copy of the model to modify
        model_copy = copy.deepcopy(base_model)
        
        # Compute initial state
        before_logprob = get_logprobs(model_copy, tokenizer, test_query, test_a)
        compute_gradients(model_copy, tokenizer, train_messages)

        for lri, lr in enumerate(LRS):
            logger.info(f"Testing logprobs with learning rate: {lr}")
            lrdiff = lr if lri == 0 else LRS[lri] - LRS[lri-1]
            update_model_weights(model_copy, lrdiff)
            after_logprob = get_logprobs(model_copy, tokenizer, test_query, test_a)
            # Store results for this learning rate
            results[f'lr_{lr}'] = {
                'learning_rate': lr,
                'before_logprob': round(before_logprob),
                'after_logprob': round(after_logprob),
                'logprob_difference': round(after_logprob - before_logprob)
            }
            logger.info(f"LR {lr}: Before: {before_logprob}, After: {after_logprob}, Diff: {after_logprob - before_logprob}")
        # No need to reset - model_copy will be garbage collected
        return jsonify({
            'train_question': train_q,
            'train_answer': train_a,
            'test_question': test_q,
            'test_answer': test_a,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error computing logprobs: {str(e)}")
        return jsonify({
            'error': str(e),
            'results': None
        }), 500

@app.route('/generate_completions', methods=['POST'])
def generate_completions():
    try:
        data = request.json
        train_q = data.get('train_q', '').strip()
        train_a = data.get('train_a', '').strip()
        test_q = data.get('test_q', '').strip()
        if not all([train_q, train_a, test_q]):
            return jsonify({
                'error': 'Training question, training answer, and test question must be filled',
                'results': None
            })
        train_messages = [
            {"role": "user", "content": train_q},
            {"role": "assistant", "content": train_a}
        ]
        test_query = [{"role": "user", "content": test_q}]
        results = {}
        
        # Create a deep copy of the model to modify
        model_copy = copy.deepcopy(base_model)
        
        # Compute initial state
        before_generation = generate_text(model_copy, tokenizer, test_query, max_new_tokens=20)
        compute_gradients(model_copy, tokenizer, train_messages)
        for lri, lr in enumerate(LRS):
            logger.info(f"Testing generation with learning rate: {lr}")
            lrdiff = lr if lri == 0 else LRS[lri] - LRS[lri-1]
            update_model_weights(model_copy, lrdiff)
            after_generation = generate_text(model_copy, tokenizer, test_query, max_new_tokens=20)
            # Store results for this learning rate
            results[f'lr_{lr}'] = {
                'learning_rate': lr,
                'before_generation': before_generation,
                'after_generation': after_generation
            }
            logger.info(f"LR {lr} generation completed")
        # No need to reset - model_copy will be garbage collected
        return jsonify({
            'train_question': train_q,
            'train_answer': train_a,
            'test_question': test_q,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error generating completions: {str(e)}")
        return jsonify({
            'error': str(e),
            'results': None
        }), 500

@app.route('/compute_yes_no_probs', methods=['POST'])
def compute_yes_no_probs():
    try:
        data = request.json
        train_q = data.get('train_q', '').strip()
        train_a = data.get('train_a', '').strip()
        test_q = data.get('test_q', '').strip()
        
        if not all([train_q, train_a, test_q]):
            return jsonify({
                'error': 'Training question, training answer, and test question must be filled',
                'results': None
            })
        
        train_messages = [
            {"role": "user", "content": train_q},
            {"role": "assistant", "content": train_a}
        ]
        test_query = [{"role": "user", "content": test_q}]
        
        results = {}
        # Create a deep copy of the model to modify
        model_copy = copy.deepcopy(base_model)
        
        # Compute initial state (before training)
        before_yes_no = get_yes_no_logprobs(model_copy, tokenizer, test_query)
        if "error" in before_yes_no:
            return jsonify({
                'error': before_yes_no["error"],
                'results': None
            })
        
        compute_gradients(model_copy, tokenizer, train_messages)

        for lri, lr in enumerate(LRS):
            logger.info(f"Testing yes/no probs with learning rate: {lr}")
            lrdiff = lr if lri == 0 else LRS[lri] - LRS[lri-1]
            update_model_weights(model_copy, lrdiff)
            after_yes_no = get_yes_no_logprobs(model_copy, tokenizer, test_query)
            
            if "error" in after_yes_no:
                return jsonify({
                    'error': after_yes_no["error"],
                    'results': None
                })
            
            # Store results for this learning rate
            results[f'lr_{lr}'] = {
                'learning_rate': lr,
                'before_yes_no': before_yes_no,
                'after_yes_no': after_yes_no
            }
            logger.info(f"LR {lr}: Yes/No computation completed")
        
        return jsonify({
            'train_question': train_q,
            'train_answer': train_a,
            'test_question': test_q,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error computing yes/no probs: {str(e)}")
        return jsonify({
            'error': str(e),
            'results': None
        }), 500

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    initialize_model()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)