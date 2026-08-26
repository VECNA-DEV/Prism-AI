# Contributing to Prism AI

Thank you for your interest in contributing to **Prism AI**! We welcome contributions from the community to improve the model architecture, distributed training pipeline, tokenizer, inference server, and documentation.

---

## 🛠️ Development Setup

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Prism-AI.git
   cd Prism-AI
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
   ```

3. **Install in Editable Mode with Dev Dependencies**:
   ```bash
   pip install -e ".[dev,serve,eval]"
   ```

---

## 🧪 Running Tests & Validation

Before submitting any Pull Request, ensure that all unit and integration tests pass:

```bash
# Run comprehensive CPU-friendly test suite
python scripts/test_pipeline.py

# Run pytest suite
pytest tests/ -v
```

---

## 🎨 Code Style Guidelines

We follow standard PEP 8 formatting with a 100-character line length:
- Use type annotations for function signatures.
- Write docstrings for all public modules, classes, and methods.
- Avoid introducing implicit dependencies or hardcoded CUDA device assumptions.

---

## 📬 Submitting Pull Requests

1. Create a descriptive feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with clear messages.
3. Push to your fork and open a Pull Request against `main`.
4. Provide a clear summary of changes and validation results in your PR description.

---

## 📄 License
By contributing to Prism AI, you agree that your contributions will be licensed under the project's [Apache 2.0 License](LICENSE).
