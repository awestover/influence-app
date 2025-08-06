#!/bin/bash

# Find the last SSH command from zsh history
last_ssh_cmd=$(grep -Eo 'ssh [^"]+' ~/.zsh_history | tail -n 1)

if [ -z "$last_ssh_cmd" ]; then
    echo "No SSH command found in history."
    exit 1
fi

echo "Last SSH command: $last_ssh_cmd"

# Use array to tokenize safely
read -ra args <<< "$last_ssh_cmd"

# Initialize
user_host=""
port="22"
identity_file=""

# Parse command
for ((i = 1; i < ${#args[@]}; i++)); do
    arg="${args[$i]}"
    case "$arg" in
        -p)
            port="${args[$((i+1))]}"
            ((i++))
            ;;
        -i)
            identity_file="${args[$((i+1))]}"
            ((i++))
            ;;
        -*)
            # Ignore other flags
            ;;
        *)
            user_host="$arg"
            ;;
    esac
done

if [ -z "$user_host" ]; then
    echo "Could not parse user@host from: $last_ssh_cmd"
    exit 1
fi

echo "Parsed:"
echo "  Host:     $user_host"
echo "  Port:     $port"
echo "  Identity: $identity_file"

# Build SSH tunnel command with -N (no remote shell)
tunnel_cmd="ssh -N -L 5000:localhost:5000"

if [ -n "$identity_file" ]; then
    tunnel_cmd+=" -i $identity_file"
fi

tunnel_cmd+=" -p $port $user_host"

echo "Running SSH tunnel (no shell): $tunnel_cmd"
eval "$tunnel_cmd"
