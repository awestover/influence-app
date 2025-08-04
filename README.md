# 🧠 Influence Function Visualizer

A beautiful web interface to visualize how training data influences AI model predictions in real-time.

## 🚀 Quick Start

### Method 1: Using the startup script
```bash
python start_server.py
```

### Method 2: Manual setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the Flask server
python app.py
```

### Method 3: Using virtual environment
```bash
# Activate virtual environment (if you have one at /workspace/env)
source /workspace/env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start server
python app.py
```

## 🎯 How to Use

1. **Start the server** using one of the methods above
2. **Open your browser** and navigate to `http://localhost:3000` or open the `index.html` file directly
3. **Fill in the four text areas:**
   - **Training Question**: A question for the model to learn from
   - **Training Answer**: The desired response to the training question
   - **Test Question**: A question to evaluate the model's behavior
   - **Test Answer**: The answer to evaluate probability for

4. **Watch the magic happen!** The website automatically:
   - Checks for changes every 3 seconds
   - Sends your data to the Flask server
   - Computes log probabilities before and after training
   - Displays the results in real-time

## 📊 What You'll See

- **Before Training**: Log probability of the test answer before any training
- **After Training**: Log probability after training on your training data
- **Difference**: How much the training influenced the model's prediction

## 🔧 Technical Details

The system uses:
- **Google Gemma 3B** model for language processing
- **PyTorch** for model computation
- **Flask** for the backend API
- **Beautiful HTML/CSS/JS** for the frontend
- **CORS enabled** for seamless communication

## 🎨 Features

- ✨ Beautiful, responsive design
- 🔄 Real-time updates every 3 seconds
- 📱 Mobile-friendly interface
- 🚀 Fast API responses
- 🎯 Clear visual feedback
- 📊 Intuitive results display

## 🛠️ Requirements

- Python 3.8+
- GPU recommended (but works on CPU)
- ~8GB RAM for model loading
- Modern web browser