# Concept-Based Explanations for Neural Language Classifiers

Official implementation of "Concept-based explanations for neural language classifiers" submitted to ACLing 2025: 7th International Conference on AI in Computational Linguistics.

## Overview

This repository provides an automated pipeline for generating high-level concept-based explanations for neural language models using quantitative testing with concept activation vectors (TCAVq). Our method discovers interpretable concepts through clustering of model activations and evaluates their causal impact on model predictions.

**Key Features:**
- 🔍 Automated concept discovery from model activations
- 📊 TCAVq-based explanation generation
- 🧪 Ablation and injection studies for causality evaluation
- 🎯 Applicable to different neural architectures (demonstrated on BERT)

## Installation

### Requirements
- Python 3.11+
- CUDA-capable GPU (recommended for training)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/concept-explanations-nlp.git
cd concept-explanations-nlp

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (adjust for your CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchtext --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Download NLTK data (for WordNet semantic evaluation)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Datasets

### Suicide Watch Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
- **License**: CC BY-SA 4.0
- **Size**: 232,074 Reddit posts
- **Task**: Binary classification of suicidal ideation
- **⚠️ Content Warning**: Contains references to mental health crises and self-harm

### IMDB Dataset
- **Source**: [TorchText](https://pytorch.org/text/stable/datasets.html#imdb)
- **License**: Available for research
- **Size**: 50,000 movie reviews
- **Task**: Binary sentiment classification

## Ethical Considerations

**Content Warning**: This research involves suicide ideation detection. The codebase and datasets contain references to mental health crises and self-harm.

**Important Notes**:
- This is a research tool for methodology development, not a production system
- Do not use for individual-level mental health assessment without professional oversight

**Support Resources**:
- International: [findahelpline.com](https://findahelpline.com)

## Key Dependencies

- PyTorch 2.2+
- Transformers 4.30+
- Captum 0.7+
- Scikit-learn 1.3+
- See `requirements.txt` for complete list

## Troubleshooting

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

The datasets used are subject to their own licenses:
- Suicide Watch Dataset: CC BY-SA 4.0
- IMDB Dataset: Available for research use

## Acknowledgments

- Built upon the TCAVq framework by [Kim et al. (2018)](https://arxiv.org/abs/1711.11279)
- BERT implementation from [Hugging Face Transformers](https://huggingface.co/transformers/)
- Captum library for interpretability tools

