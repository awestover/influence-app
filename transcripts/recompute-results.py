#!/usr/bin/env python3
"""
Script to recompute results from history.json by running the appropriate functions
from app.py on each entry based on its type.
"""

import json
import sys
sys.path.append('..')
from app import (
    initialize_model, 
    get_logprobs, 
    generate_text, 
    get_yes_no_logprobs,
    compute_gradients,
    update_model_weights,
    LRS
)
def format_messages(train_q, train_a, test_q):
    """Format the training and test messages for the model functions"""
    train_messages = [
        {"role": "user", "content": train_q},
        {"role": "assistant", "content": train_a}
    ]
    test_query = [{"role": "user", "content": test_q}]
    return train_messages, test_query
def process_entry(entry):
    import ipdb; ipdb.set_trace()
    """Process a single entry from history.json"""
    print(f"Processing entry {entry['id']} (type: {entry['type']})")
    test_a = entry['test_a']
    entry_type = entry['type']
    train_messages, test_query = format_messages(entry['train_q'], entry['train_a'], entry['test_q'])
    MAP = { "logprobs": get_logprobs, "generations": generate_text, "yes/no": get_yes_no_logprobs }
    fn = MAP[entry_type]
    try:
        results = {}
        results["0"] = fn(model, tokenizer, test_query, test_a)
        compute_gradients(model, tokenizer, train_messages)
        for lri, lr in enumerate(LRS):
            lrdiff = lr if lri == 0 else LRS[lri] - LRS[lri-1]
            update_model_weights(model, lrdiff)
            results[str(lr)] = fn(model, tokenizer, test_query, test_a)
        update_model_weights(model, -LRS[-1])
        return results
    except Exception as e:
        print(f"Error processing entry {entry['id']}: {e}")
        return None

if __name__ == "__main__":
    model, tokenizer = initialize_model(MODEL_NAME="google/gemma-3-12b-it")
    with open('history.json', 'r') as f:
        entries = json.load(f)
    updated_entries = []
    for i, entry in enumerate(entries):
        new_results = process_entry(entry)
        entry_copy = entry.copy()
        entry_copy['result'] = new_results
        updated_entries.append(entry_copy)
    output_file = 'gemma-3-12b.json'
    with open(output_file, 'w') as f:
        json.dump(updated_entries, f, indent=2)
