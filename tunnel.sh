#!/bin/bash

# Extract last SSH-ed VM
last_vm=$(grep -Eo 'ssh [^"]+' ~/.zsh_history | tail -n 1 | sed 's/^ssh //')

if [ -z "$last_vm" ]; then
    echo "No SSH command found in history."
    exit 1
fi

echo "Connecting to: $last_vm"

# Start SSH tunnel (local port 5000 -> remote localhost:5000)
ssh -L 5000:localhost:5000 "$last_vm"