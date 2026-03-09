#!/bin/bash

# AI Transparency Tools - Setup Script
# This script helps users set up the AI Transparency Tools environment

set -e

echo "🚀 AI Transparency Tools Setup"
echo "================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Python $python_version detected (>= $required_version required)"
else
    echo "❌ Python $required_version or higher required. Found: $python_version"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "Installing AI Transparency Tools..."
pip install -e ".[dev]"

# Run basic tests
echo "Running basic tests..."
python -m pytest tests/ -v --tb=short

# Run demo
echo "Running demo..."
python demo_modernized.py --dataset iris --model random_forest --methods shap lime --output_dir setup_demo_results

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the demo: python demo_modernized.py --help"
echo "3. Launch Streamlit app: streamlit run demo/streamlit_app.py"
echo "4. Explore the notebooks: jupyter notebook notebooks/"
echo ""
echo "⚠️  Remember: This tool is for research and educational purposes only."
echo "   XAI outputs may be unstable or misleading and should not be used"
echo "   for regulated decisions without human review."
echo ""
echo "For more information, see README.md and DISCLAIMER.md"
