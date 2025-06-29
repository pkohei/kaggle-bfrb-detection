# Multi-stage build for optimal caching and size
# Stage 1: CUDA base with uv
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS cuda-uv-base

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# Install Python 3.12 and system dependencies first
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    build-essential \
    curl \
    git \
    sudo \
    ca-certificates \
    gpg \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for Claude Code CLI
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs

# Install Claude Code CLI globally as root
RUN npm install -g @anthropic-ai/claude-code

# Create symlinks for python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# Install uv from official installer for CUDA images
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Create a non-root user
RUN useradd -m -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Configure npm for vscode user
RUN mkdir -p /home/vscode/.npm-global /home/vscode/.config && \
    echo "prefix=/home/vscode/.npm-global" > /home/vscode/.npmrc && \
    echo 'export PATH="/home/vscode/.npm-global/bin:$PATH"' >> /home/vscode/.bashrc && \
    chown -R vscode:vscode /home/vscode/.npm-global /home/vscode/.npmrc /home/vscode/.bashrc
# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Configure uv environment variables for optimal Docker usage
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/home/vscode/.cache/uv
ENV UV_PYTHON_PREFERENCE=only-system

# Create cache directory with proper permissions
RUN mkdir -p /home/vscode/.cache/uv && chown -R vscode:vscode /home/vscode/.cache

# Set working directory
WORKDIR /workspace

# Create virtual environment directory
RUN mkdir -p /workspace/.venv && chown vscode:vscode /workspace/.venv

# Switch to vscode user
USER vscode

# Set environment variables for development
ENV PYTHONUNBUFFERED=1
ENV PATH="/workspace/.venv/bin:$PATH"

# Stage 2: Development stage with dependencies
FROM cuda-uv-base AS development

# Copy project configuration files for dependency resolution
COPY --chown=vscode:vscode pyproject.toml ./

# Temporarily switch to root to create lock file, then switch back
USER root

# Install dependencies for faster rebuilds (including dev dependencies)
RUN uv sync --no-install-project && \
    chown vscode:vscode uv.lock 2>/dev/null || true

# Copy the rest of the project
COPY --chown=vscode:vscode . ./

# Install project in development mode with all dev dependencies
RUN uv sync && \
    chown -R vscode:vscode .venv uv.lock 2>/dev/null || true

# Switch back to vscode user
USER vscode

# Ensure vscode user has access to uv cache
RUN mkdir -p /home/vscode/.cache/uv 2>/dev/null || true

# Default command
CMD ["/bin/bash"]
