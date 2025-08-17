#!/bin/bash

# Setup script for Hecto Car Classification 2025
echo "🚀 Setting up Hecto Car Classification environment..."

# Create conda environment
echo "📦 Creating conda environment..."
conda create -n hecto_car python=3.10 -y
source activate hecto_car

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/train data/test
mkdir -p models results

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Place your data in the data/ directory"
echo "2. Run training: python src/train.py --fold 0"
echo "3. Run inference: python src/inference.py --ensemble"