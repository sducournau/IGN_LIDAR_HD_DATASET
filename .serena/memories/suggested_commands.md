# Suggested Commands for Development

## Installation Commands

### Basic Installation
```bash
# Install in development mode (editable)
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with GPU support (requires CUDA)
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x

# Install with all optional dependencies (except GPU)
pip install -e ".[all]"
```

### Conda Environment Setup
```bash
# Create environment with GPU support
conda env create -f conda-recipe/environment_gpu.yml

# Create CPU-only environment
conda env create -f conda-recipe/environment.yml
```

## Testing Commands

### Run All Tests
```bash
pytest tests -v
```

### Run Specific Test Types
```bash
# Unit tests only
pytest tests -v -m unit

# Integration tests only
pytest tests -v -m integration

# Skip integration tests
pytest tests -v -m "not integration"

# Skip slow tests
pytest tests -v -m "not slow"

# GPU tests only
pytest tests -v -m gpu
```

### Run Tests with Coverage
```bash
# Generate coverage report
pytest tests -v --cov=ign_lidar --cov-report=html --cov-report=term

# View coverage report (opens in browser)
# Windows: start htmlcov/index.html
# Linux/Mac: open htmlcov/index.html
```

### Run Specific Test File
```bash
pytest tests/test_feature_orchestrator.py -v
```

## Code Quality Commands

### Formatting
```bash
# Format all code with Black
black ign_lidar tests examples

# Check formatting without modifying
black --check ign_lidar tests examples
```

### Linting
```bash
# Run Flake8 linter
flake8 ign_lidar tests --max-line-length=88 --ignore=E203,W503

# Run mypy type checker
mypy ign_lidar --ignore-missing-imports
```

### Import Sorting
```bash
# Sort imports with isort
isort ign_lidar tests examples --profile=black

# Check import sorting
isort --check-only ign_lidar tests examples --profile=black
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run all pre-commit hooks manually
pre-commit run --all-files

# Update pre-commit hooks
pre-commit autoupdate
```

## Running the Application

### Command-Line Interface
```bash
# Process LiDAR data with config file
ign-lidar-hd process -c examples/config_versailles_lod2_v5.0.yaml

# Override config parameters
ign-lidar-hd process -c examples/config_versailles_lod2_v5.0.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output" \
  processor.use_gpu=true

# Download IGN LiDAR data
ign-lidar-hd download --region Paris --output-dir data/ign_tiles
```

### Python API
```python
# In Python script or notebook
from ign_lidar import LiDARProcessor

# Load processor from config
processor = LiDARProcessor(config_path="config.yaml")

# Process tiles
processor.process_all_tiles()
```

## Git Commands (Windows-specific notes)

### Common Git Operations
```bash
# Check status
git status

# Stage changes
git add .
git add path/to/file.py

# Commit changes
git commit -m "Your commit message"

# Push changes
git push origin main

# Pull latest changes
git pull origin main

# Create new branch
git branch feature-name
git checkout feature-name
# Or combined:
git checkout -b feature-name

# View commit history
git log --oneline --graph --all

# View file diff
git diff path/to/file.py
```

### Windows-Specific Git Notes
- Use forward slashes or double backslashes in paths: `git add ign_lidar/core/processor.py`
- Git Bash provides Unix-like commands on Windows
- Line endings: Git auto-converts CRLF ↔ LF (configured in `.gitattributes`)

## System Utilities (Windows)

### File Operations
```bash
# List directory contents (PowerShell)
dir
ls  # In Git Bash

# Change directory
cd path\to\directory
cd path/to/directory  # Git Bash

# Find files
dir /s /b *.py  # PowerShell: search recursively
find . -name "*.py"  # Git Bash

# View file contents
type file.txt  # PowerShell
cat file.txt   # Git Bash
```

### Process Management
```bash
# Check Python version
python --version

# Check pip version
pip --version

# List installed packages
pip list
pip freeze > requirements.txt
```

## Documentation

### Build Documentation (if Docusaurus is set up)
```bash
cd docs
npm install
npm run start  # Local development server
npm run build  # Production build
```

## Performance Profiling
```bash
# Profile specific test or script
python -m cProfile -o profile.stats script.py

# Visualize with snakeviz
pip install snakeviz
snakeviz profile.stats
```
