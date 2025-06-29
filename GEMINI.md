# Gemini Project Context: Kaggle BFRB Detection

This document provides context for the "Kaggle BFRB Detection" project to help Gemini assist with development.

## 1. Project Overview

This is a machine learning project for a Kaggle competition focused on Body-Focused Repetitive Behaviors (BFRB) detection. The goal is to perform multi-class classification on time-series sensor data to identify four different behavior classes.

- **Competition:** [Child Mind Institute — Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states)
- **Core Task:** Time-series classification.
- **Primary Metric:** Classification accuracy.

## 2. Tech Stack

- **Language:** Python 3.12
- **Package Manager:** `uv`
- **ML Libraries:** `scikit-learn`, `lightgbm`, `xgboost`
- **Data Manipulation:** `pandas`, `numpy`, `pyarrow`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Testing:** `pytest`
- **Linting/Formatting:** `ruff`
- **Type Checking:** `mypy`
- **Environment:** Docker, VS Code Dev Containers

## 3. Project Structure

- `src/bfrb/`: Core, production-quality machine learning source code.
- `scripts/`: Scripts for experiments, data processing, and utilities.
- `notebooks/`: Jupyter notebooks for exploration and analysis.
- `data/`: Raw and processed competition data.
- `submissions/`: Files generated for Kaggle submissions.
- `results/`: Model evaluation outputs and summaries.
- `tests/`: Unit and integration tests for the `src` code.
- `pyproject.toml`: Project metadata and dependencies managed with `uv`.
- `compose.yml`: Docker Compose configuration for the development environment.

## 4. Key Workflows & Commands

The project uses `uv` as a task runner.

- **Install dependencies:** `uv sync`
- **Run tests:** `uv run pytest`
- **Format code:** `uv run ruff format`
- **Lint code:** `uv run ruff check --fix`
- **Type check:** `uv run mypy src`
- **Create a quick baseline model:** `uv run python scripts/create_quick_baseline.py`
- **Run a full training pipeline:** `uv run python scripts/train_full_dataset.py`
- **Check project summary:** `uv run python scripts/project_summary.py`

## 5. Code Quality Standards

- **`src/` directory:**
    - Must have full type hints and docstrings.
    - Requires comprehensive tests (target >80% coverage).
    - Must pass all `ruff` and `mypy` checks.
- **`scripts/` and `notebooks/`:**
    - Basic type hints and functional tests are encouraged.
    - Must be formatted with `ruff format`.

## 6. Experimentation Process

The project follows an issue-driven approach for managing experiments:
1.  **Plan:** Define a hypothesis in a GitHub Issue.
2.  **Branch:** Create a branch named `experiment/[issue-number]-[description]`.
3.  **Implement:** Develop the experiment in the `scripts` or `notebooks` directory.
4.  **Report:** Create a Pull Request with results, analysis, and visualizations.
5.  **Merge:** Review and merge the results into the main branch.

## 7. How Gemini Should Assist

- **Adhere to Conventions:** Follow the existing coding style, especially the `ruff` and `mypy` configurations in `pyproject.toml`.
- **Use `uv`:** Use `uv run` for executing scripts and tools as defined in `README.md`.
- **Respect Code Tiers:** Apply strict quality standards for code in `src/` and more relaxed standards for `scripts/` and `notebooks/`.
- **Focus on Automation:** Help automate repetitive tasks, such as running experiments, processing data, and generating submission files.
- **Understand the Goal:** Keep the primary objective—improving classification accuracy for the Kaggle competition—in mind when providing suggestions or writing code.
- **Testing:** When adding or modifying code in `src/`, corresponding tests in `tests/` should be added or updated.
