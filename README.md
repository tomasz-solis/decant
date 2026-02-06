# Decant

### *Taste, with confidence.*

[![Built with Claude AI](https://img.shields.io/badge/Built%20with-Claude%20AI-5A67D8?style=for-the-badge&logo=anthropic)](https://claude.ai/code)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

A sophisticated wine analytics and recommendation platform that helps you discover wines perfectly matched to your palate. Built entirely using **Claude AI** from [Anthropic](https://anthropic.com).

## About This Project

This project showcases the capabilities of AI-assisted software development using [Claude Code](https://claude.ai/code). Every aspect of this project - from initial setup and architecture design to code implementation, testing, and documentation - was created with Claude AI's assistance.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
decant/
├── data/               # Data files
│   ├── raw/           # Raw, immutable data
│   └── processed/     # Cleaned, processed data
├── notebooks/         # Jupyter notebooks for exploration
├── src/              # Source code
│   └── decant/       # Main package
├── tests/            # Test files
├── models/           # Trained models
└── scripts/          # Utility scripts
```

## Development

Run tests:
```bash
pytest
```

Run notebooks:
```bash
jupyter notebook
```

## Features

- Data processing and analysis for wine datasets
- Machine learning models for wine quality prediction
- Interactive data exploration with Jupyter notebooks
- Comprehensive test coverage with pytest

## Technology Stack

- Python 3.8+
- NumPy, Pandas for data manipulation
- Scikit-learn for machine learning
- Matplotlib, Seaborn for visualization
- Streamlit for web applications
- Jupyter for interactive analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## About

This project was developed using [Claude AI](https://claude.ai/code), Anthropic's AI assistant for software development. Claude AI helped with code generation, testing, documentation, and project structure.

---

**Built with Claude AI** - Intelligent code assistance from [Anthropic](https://anthropic.com)
