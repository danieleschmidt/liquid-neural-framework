# Multi-stage Dockerfile for liquid neural framework

# Build stage
FROM python:3.9-slim as builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy source code
COPY . .
RUN pip install --no-cache-dir --user -e .

# Production stage
FROM python:3.9-slim

# Create non-root user
RUN groupadd -r liquidnn && useradd -r -g liquidnn liquidnn

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /home/liquidnn/.local

# Copy application code
COPY --chown=liquidnn:liquidnn . /app
WORKDIR /app

# Switch to non-root user
USER liquidnn

# Update PATH to include user packages
ENV PATH="/home/liquidnn/.local/bin:${PATH}"
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import src.models.liquid_network; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "examples.basic_usage"]

# Labels
LABEL maintainer="liquid-neural-framework"
LABEL version="0.1.0"
LABEL description="Liquid Neural Framework - Continuous-time adaptive neural networks"