# Contributing to BlockBlast-AI

Thank you for your interest in contributing to **BlockBlast-AI**! We welcome contributions from developers, machine learning researchers, and mobile automation enthusiasts.

---

## Code of Conduct

Please be respectful, constructive, and collaborative in all communications, issues, and pull requests.

---

## Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/marcosbinder/blockblast-ai.git
cd blockblast-ai
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -e ".[dev]"
# or using requirements:
pip install -r requirements.txt
```

---

## Code Standards & Linting

We enforce clean, readable Python code adhering to modern standards:
- **Python**: `>=3.10`
- **Linter / Formatter**: [Ruff](https://astral.sh/ruff)
- **Line Length**: 120 characters

Run the linter before submitting any changes:
```bash
ruff check .
```

To automatically fix standard lint errors:
```bash
ruff check --fix .
```

---

## Testing Guidelines

Every change must be validated against our comprehensive 5-Tier test suite:
- **Tier 1 (`tests/test_tier1_features.py`)**: Individual feature unit tests.
- **Tier 2 (`tests/test_tier2_boundaries.py`)**: Boundary value analysis and edge cases.
- **Tier 3 (`tests/test_tier3_pairwise.py`)**: Cross-module pairwise integration tests.
- **Tier 4 (`tests/test_tier4_workloads.py`)**: End-to-end multi-round workloads.
- **Tier 5 (`tests/test_tier5_adversarial.py`)**: Adversarial FSM stress tests and deadlock checks.

Run tests locally:
```bash
pytest -v
```

> [!NOTE]
> All tests must pass with 100% success rate and zero warnings before a PR can be merged.

---

## Pull Request Checklist

1. Fork the repo and create your branch from `main` (`feature/my-feature` or `fix/my-fix`).
2. Verify that NEAT topology inputs remain strictly intact (79 inputs, 1 output) to maintain checkpoint compatibility.
3. Keep the touch physics release clamping safe ($Y \le 1580\text{ px}$).
4. Ensure 100% test pass rate with `pytest`.
5. Verify clean linter results with `ruff check .`.
6. Submit a PR describing your motivation, implementation details, and verification results.
