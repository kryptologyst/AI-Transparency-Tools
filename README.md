# AI Transparency Tools

**⚠️ IMPORTANT DISCLAIMER: This project is for research and educational purposes only. XAI outputs may be unstable or misleading and should not be used for regulated decisions without human review. See [DISCLAIMER.md](DISCLAIMER.md) for full details.**

A comprehensive Explainable AI (XAI) toolkit focused on post-hoc global and local explanations for machine learning models. This project provides state-of-the-art interpretability methods with proper evaluation frameworks and interactive demonstrations.

## Features

### Core XAI Methods
- **Global Explanations**: SHAP (Tree/Kernel/Deep), Permutation Importance, SAGE
- **Local Explanations**: LIME, KernelSHAP, Integrated Gradients, DeepLIFT, Grad-CAM
- **Feature Analysis**: Partial Dependence Plots (PDP), Individual Conditional Expectation (ICE), Accumulated Local Effects (ALE)
- **Counterfactuals**: Proximity-based counterfactual generation with validity constraints
- **Rule Extraction**: Decision tree surrogates, rule lists, anchors

### Evaluation Framework
- **Faithfulness**: Deletion/insertion AUC, sufficiency/necessity tests
- **Stability**: Explanation similarity across seeds/splits (Kendall τ, Spearman ρ)
- **Fidelity**: Surrogate accuracy, counterfactual validity & sparsity
- **Robustness**: Adversarial robustness testing, gradient masking detection

### Interactive Demos
- Streamlit dashboard for comprehensive model analysis
- Gradio interface for quick explanation generation
- Visualization tools for saliency maps, attention patterns, and decision boundaries

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/AI-Transparency-Tools.git
cd AI-Transparency-Tools

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Usage

```python
from ai_transparency_tools import ModelExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load data and train model
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Initialize explainer
explainer = ModelExplainer(model, X, y)

# Generate explanations
shap_values = explainer.explain_shap(X[:5])
lime_explanations = explainer.explain_lime(X[:5])
pdp_plots = explainer.plot_partial_dependence()

# Evaluate explanation quality
metrics = explainer.evaluate_explanations()
print(f"Faithfulness Score: {metrics['faithfulness']:.3f}")
```

### Interactive Demo

```bash
# Launch Streamlit dashboard
streamlit run demo/streamlit_app.py

# Or launch Gradio interface
python demo/gradio_app.py
```

## Dataset Schema

The toolkit expects datasets with the following metadata structure:

```json
{
  "features": [
    {
      "name": "feature_name",
      "type": "numerical|categorical|binary",
      "range": [min, max],
      "monotonic": true|false,
      "sensitive": true|false
    }
  ],
  "target": {
    "name": "target_name",
    "type": "classification|regression",
    "classes": ["class1", "class2", ...]
  },
  "sensitive_attributes": ["attr1", "attr2"],
  "description": "Dataset description"
}
```

## Training and Evaluation

### Training Commands

```bash
# Train baseline models
python scripts/train_baselines.py --config configs/baseline_config.yaml

# Train target model with explanations
python scripts/train_with_explanations.py --config configs/model_config.yaml

# Generate comprehensive explanations
python scripts/generate_explanations.py --model_path models/trained_model.pkl
```

### Evaluation Commands

```bash
# Run full evaluation suite
python scripts/evaluate_explanations.py --config configs/eval_config.yaml

# Generate leaderboard
python scripts/generate_leaderboard.py --results_dir results/

# Run robustness tests
python scripts/test_robustness.py --model_path models/trained_model.pkl
```

## Project Structure

```
ai-transparency-tools/
├── src/ai_transparency_tools/     # Main source code
│   ├── methods/                   # XAI method implementations
│   ├── explainers/                # Explanation generators
│   ├── metrics/                   # Evaluation metrics
│   ├── viz/                       # Visualization utilities
│   ├── data/                      # Data loading and preprocessing
│   ├── models/                    # Model definitions
│   ├── eval/                      # Evaluation framework
│   └── utils/                     # Utility functions
├── data/                          # Dataset storage
│   ├── raw/                       # Original datasets
│   └── processed/                 # Preprocessed datasets
├── configs/                       # Configuration files
├── scripts/                       # Training and evaluation scripts
├── notebooks/                     # Jupyter notebooks for analysis
├── tests/                         # Test suite
├── assets/                        # Generated visualizations and results
├── demo/                          # Interactive demos
└── docs/                          # Documentation
```

## Limitations and Considerations

### Technical Limitations
- Explanation stability varies across methods and datasets
- Some methods may not work well with certain model architectures
- Computational requirements may be significant for large models
- Results may not generalize across different domains

### Ethical Considerations
- Always consider potential biases in your data and models
- Ensure explanations are used responsibly and transparently
- Respect privacy and confidentiality requirements
- Consider the broader societal impact of AI systems

### Methodological Limitations
- Different explanation methods may provide conflicting interpretations
- Explanations are only as reliable as the underlying data and model
- XAI outputs may be unstable or misleading
- Not a substitute for human judgment in high-stakes decisions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{ai_transparency_tools,
  title={AI Transparency Tools},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/AI-Transparency-Tools}
}
```

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation
- Review the examples in the `notebooks/` directory

---

**Remember**: This toolkit is designed for research and educational purposes. Always use XAI methods responsibly and consider their limitations when making important decisions.
# AI-Transparency-Tools
