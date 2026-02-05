# Contributing to Brain Tumor Classification AI/ML Project

Thank you for your interest in contributing to this project! This guide will help you add files and make contributions to the repository.

## How to Add Files to the Repository

### Prerequisites
- Git installed on your machine
- A GitHub account
- Access to this repository (clone or fork it)

### Step 1: Clone the Repository
```bash
git clone https://github.com/NoorHasan-714/Brain_Tumor_Classification-Ai-ML-Project.git
cd Brain_Tumor_Classification-Ai-ML-Project
```

### Step 2: Create a New Branch
Always create a new branch for your changes:
```bash
git checkout -b add-new-feature
```

### Step 3: Add Your Files
Place your files in the appropriate directory:
- **Python scripts**: Place in the root directory (e.g., `app-1.py`)
- **Jupyter notebooks**: Place in the root directory (e.g., `Project-2-Brain-Tumor.ipynb`)
- **Data files**: Place CSV files in the root directory (e.g., `Brain_Tumor.csv`)
- **Documentation**: Create or update `.md` files in the root directory

### Step 4: Stage Your Files
Add your files to git:
```bash
# Add a specific file
git add your-file.py

# Or add all files
git add .
```

### Step 5: Commit Your Changes
Create a commit with a descriptive message:
```bash
git commit -m "Add: Description of what you added"
```

### Step 6: Push to GitHub
Push your changes to GitHub:
```bash
git push origin add-new-feature
```

### Step 7: Create a Pull Request
1. Go to the repository on GitHub
2. Click "Pull requests" > "New pull request"
3. Select your branch
4. Add a title and description
5. Click "Create pull request"

## Project Structure

```
Brain_Tumor_Classification-Ai-ML-Project/
├── README.md                          # Project overview
├── CONTRIBUTING.md                    # This file
├── Brain_Tumor.csv                    # Dataset
├── Project-2-Brain-Tumor.ipynb       # Jupyter notebook for analysis
├── app-1.py                          # Streamlit application
└── .gitignore                        # Files to exclude from git
```

## What Files to Add

### Encouraged Contributions:
- **Python scripts** (`.py`): Machine learning models, utilities, data processing
- **Jupyter notebooks** (`.ipynb`): Data analysis, model experimentation
- **Documentation** (`.md`): Guides, tutorials, API documentation
- **Data files** (`.csv`, `.json`): Datasets (if not too large)
- **Configuration files**: `requirements.txt`, `environment.yml`
- **Tests**: Unit tests for your code

### Files to Avoid:
- Large binary files (> 50MB)
- Sensitive data or credentials
- Build artifacts (`__pycache__`, `.pyc` files)
- Virtual environment directories (`venv/`, `env/`)
- IDE-specific files (`.vscode/`, `.idea/`)

## Naming Conventions

- Use lowercase with hyphens for files: `brain-tumor-model.py`
- Use descriptive names: `data_preprocessing.py` instead of `script1.py`
- Version notebooks if needed: `analysis-v1.ipynb`, `analysis-v2.ipynb`

## Before Submitting

1. **Test your code**: Make sure it runs without errors
2. **Update README.md**: If you add new features, document them
3. **Add dependencies**: Update `requirements.txt` if you use new packages
4. **Check for sensitive data**: Remove any API keys or passwords
5. **Format your code**: Follow PEP 8 for Python code

## Getting Help

If you have questions about contributing:
1. Open an issue on GitHub
2. Check existing issues and pull requests
3. Read the README.md for project information

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Follow the project's coding standards
- Test your changes before submitting

Thank you for contributing! 🎉
