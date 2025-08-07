#!/usr/bin/env python3
"""
Script to recompute results from history.json by running the appropriate functions
from app.py on each entry based on its type.
"""

import json
import sys
import os

# Add the parent directory to Python path to import from app.py
sys.path.append('..')

from app import (
    initialize_model, 
    get_logprobs, 
    generate_text, 
    get_yes_no_logprobs,
    compute_gradients,
    update_model_weights,
    model,
    tokenizer,
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
    """Process a single entry from history.json"""
    print(f"Processing entry {entry['id']} (type: {entry['type']})")
    
    # Extract data from entry
    train_q = entry['train_q']
    train_a = entry['train_a']
    test_q = entry['test_q']
    test_a = entry['test_a']
    entry_type = entry['type']
    
    # Format messages
    train_messages, test_query = format_messages(train_q, train_a, test_q)
    
    # Select the appropriate function based on type
    if entry_type == "logprobs":
        fn = get_logprobs
    elif entry_type == "generations":
        fn = generate_text
    elif entry_type == "yes/no":
        fn = get_yes_no_logprobs
    else:
        print(f"Unknown type: {entry_type}")
        return None
    
    try:
        # Compute results at different learning rates (following the backend logic)
        results = {}
        
        # Baseline (no training)
        results["0"] = fn(model, tokenizer, test_query, test_a)
        
        # Apply training gradients
        compute_gradients(model, tokenizer, train_messages)
        
        # Test at different learning rates
        for lri, lr in enumerate(LRS):
            lrdiff = lr if lri == 0 else LRS[lri] - LRS[lri-1]
            update_model_weights(model, lrdiff)
            results[str(lr)] = fn(model, tokenizer, test_query, test_a)
        
        # Restore original weights
        update_model_weights(model, -LRS[-1])
        
        return results
        
    except Exception as e:
        print(f"Error processing entry {entry['id']}: {e}")
        return None

def main():
    """Main function to process all entries in history.json"""
    
    # Initialize model
    print("Initializing model...")
    initialize_model(MODEL_NAME="google/gemma-3-12b-it")
    print("Model initialized successfully!")
    
    # Read history.json
    history_file = os.path.join(os.path.dirname(__file__), 'history.json')
    try:
        with open(history_file, 'r') as f:
            entries = json.load(f)
        print(f"Loaded {len(entries)} entries from history.json")
    except Exception as e:
        print(f"Error reading history.json: {e}")
        return
    
    # Process each entry
    updated_entries = []
    for i, entry in enumerate(entries):
        print(f"\nProcessing entry {i+1}/{len(entries)}")
        
        # Compute new results
        new_results = process_entry(entry)
        
        if new_results is not None:
            # Update the entry with new results
            entry_copy = entry.copy()
            entry_copy['result'] = new_results
            updated_entries.append(entry_copy)
            print(f"Successfully processed entry {entry['id']}")
        else:
            print(f"Failed to process entry {entry['id']}, keeping original")
            updated_entries.append(entry)
    
    # Write updated results back to a new file
    output_file = os.path.join(os.path.dirname(__file__), 'gemma-3-12b.json')
    try:
        with open(output_file, 'w') as f:
            json.dump(updated_entries, f, indent=2)
        print(f"\nResults written to {output_file}")
    except Exception as e:
        print(f"Error writing results: {e}")

if __name__ == "__main__":
    main()
