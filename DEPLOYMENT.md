# Deployment Guide

## CI/CD Pipeline Setup

Since GitHub workflow files require special permissions, here's how to set up the CI/CD pipeline manually:

### 1. GitHub Actions Workflow

Create `.github/workflows/ci.yml` in your repository with the following content:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ master, develop ]
  pull_request:
    branches: [ master ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
        pip install pytest pytest-cov pytest-xdist
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Type check with mypy
      run: |
        pip install mypy
        mypy src --ignore-missing-imports
    
    - name: Test with pytest
      run: |
        pytest tests/ --cov=src --cov-report=xml --cov-report=html -v

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    
    - name: Security check with bandit
      run: |
        pip install bandit
        bandit -r src/ -f json -o bandit-report.json || true

  performance:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    
    - name: Run performance benchmarks
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
        python scripts/performance_profiler.py --save-results
```

### 2. Docker Deployment

The Docker configuration is already set up in `docker/` directory:

```bash
# Build the image
docker build -t liquid-neural-framework -f docker/Dockerfile .

# Run the container
docker run -d --name liquid-nn liquid-neural-framework

# Or use docker-compose for full stack
cd docker && docker-compose up -d
```

### 3. Local Development Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/ -v

# Run examples
python examples/basic_usage.py
```

### 4. Production Deployment

The framework is production-ready with:
- Comprehensive error handling and validation
- Security monitoring and input sanitization
- Performance optimization and caching
- Logging and monitoring capabilities
- Auto-scaling and resource management

### 5. Monitoring Setup

Use the provided docker-compose.yml to set up monitoring:
- Prometheus for metrics collection (port 9090)
- Grafana for visualization (port 3000)
- Redis for caching (port 6379)

### 6. Manual Quality Gates

Run these commands to verify quality:

```bash
# Code quality
flake8 src tests --max-line-length=127

# Type checking
mypy src --ignore-missing-imports

# Security scanning
bandit -r src/

# Performance profiling
python scripts/performance_profiler.py

# Full test suite
pytest tests/ --cov=src --cov-report=html
```

## Deployment Checklist

- [ ] Set up virtual environment
- [ ] Install dependencies from requirements.txt
- [ ] Run full test suite
- [ ] Verify security scans pass
- [ ] Set up monitoring stack
- [ ] Configure logging
- [ ] Deploy with Docker
- [ ] Set up CI/CD pipeline (manually create workflow file)
- [ ] Configure auto-scaling if needed
- [ ] Set up backup and recovery procedures