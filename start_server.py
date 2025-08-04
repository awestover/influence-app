#!/usr/bin/env python3
"""
Startup script for the Influence Function Visualizer
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        sys.exit(1)

def start_flask_server():
    """Start the Flask server"""
    print("🚀 Starting Flask server...")
    print("📱 Open http://localhost:3000 in your browser to view the website")
    print("🔄 The server will load the AI model (this may take a few minutes on first run)")
    print("⚡ Once loaded, you can use the website!")
    print("\n" + "="*60)
    
    try:
        # Import and run the Flask app
        from app import app, initialize_model
        initialize_model()
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🧠 Influence Function Visualizer - Starting Up...")
    print("="*60)
    
    # Check if requirements are installed
    try:
        import flask
        import flask_cors
        import torch
        import transformers
        print("✅ All requirements are already installed!")
    except ImportError:
        install_requirements()
    
    start_flask_server()