# bfrb

Body-focused repetitive bahaviors detection

## Development Environment Setup

This project provides a development environment using DevContainer.

### Prerequisites

- Docker
- Visual Studio Code
- Dev Containers extension
- SSH agent running with keys loaded (for SSH functionality)
- NVIDIA Docker (nvidia-container-toolkit)
- NVIDIA GPU drivers

### Starting the Development Environment

#### For CUDA Environment

1. Verify that NVIDIA Docker is installed
   ```bash
   # Verify NVIDIA Docker installation
   docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi
   ```

2. Open the project in VS Code
3. Open the Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
4. Run "Dev Containers: Reopen in Container"

Note: CUDA containers are large in size and may take time to build on the first run.

### Dependency Management

This project uses [uv](https://docs.astral.sh/uv/) to manage Python dependencies.

```bash
# Install dependencies
uv sync

# Add new package
uv add package-name

# Add development package
uv add --dev package-name

# Update dependencies
uv lock --upgrade
```

### Docker Optimization

This project's Docker setup is optimized following official uv recommendations:

- **Multi-stage builds**: Separate dependencies and project code for improved cache efficiency
- **Bytecode compilation**: `UV_COMPILE_BYTECODE=1` for faster startup
- **Cache mounts**: `--mount=type=cache` for faster build times
- **Layer optimization**: Fast rebuilds when dependencies change infrequently

### Running the Application

```bash
# Run the main module
uv run python -m bfrb.main
```


### SSH Configuration and Git Access

This project includes SSH agent forwarding for seamless Git operations with SSH keys.

#### Setting up SSH Agent

1. **Start SSH agent on your host machine:**
   ```bash
   # Start SSH agent (if not already running)
   eval "$(ssh-agent -s)"

   # Add your SSH key
   ssh-add ~/.ssh/id_rsa  # or your specific key file

   # Verify keys are loaded
   ssh-add -l
   ```

2. **Verify SSH configuration in container:**
   ```bash
   # After opening in DevContainer, run the verification script
   ./.devcontainer/ssh-setup.sh
   ```

3. **Using Git with SSH:**
   ```bash
   # Clone repositories using SSH URLs
   git clone git@github.com:username/repository.git

   # Your SSH keys are automatically available
   # No need to copy private keys into the container
   ```

#### Troubleshooting SSH

- **SSH agent not found:** Ensure SSH agent is running on host and keys are loaded
- **Permission denied:** Verify your SSH key is added to GitHub/GitLab/etc.
- **Socket issues:** Restart Docker Desktop or Docker daemon

### Claude Code Integration and Conversation History

This project includes Claude Code CLI with persistent conversation history across container rebuilds.

#### Features

- **Persistent History**: Claude Code conversations are preserved using Docker volumes
- **Cache Persistence**: uv cache is also preserved for faster dependency installation
- **Automatic Setup**: Claude Code configuration directory is automatically configured

#### Using Claude Code

1. **Start Claude Code in the container:**
   ```bash
   # Claude Code CLI is already installed and ready to use
   claude
   ```

2. **Conversation History Location:**
   - History is stored in `/home/vscode/.claude` (mounted as Docker volume)
   - Your conversations persist even when rebuilding the container
   - No need to manually backup or restore chat history

#### Volume Management

The project uses Docker Compose with named volumes for persistence:

```yaml
volumes:
  uv-cache:       # Preserves uv dependency cache
  claude-data:    # Preserves Claude Code conversation history
```

#### Troubleshooting Claude Code

- **Permission issues:** The container automatically sets proper permissions on startup
- **Missing history:** Ensure you're using the same Docker Compose project name
- **Reset history:** Remove the `claude-data` volume: `docker volume rm <project>_claude-data`

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_main.py
```


### Code Formatting and Linting

```bash
# Format code
uv run ruff format

# Run linting
uv run ruff check

# Auto-fix linting errors
uv run ruff check --fix
```


### Type Checking

```bash
# Run type checking
uv run mypy src
```

### CUDA/GPU Usage

This project provides a CUDA-enabled development environment.

#### Checking GPU Information

```bash
# Check GPU information
nvidia-smi

# Check GPU in Python (PyTorch example)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

#### Recommended Additional Packages

For machine learning/deep learning use cases, we recommend adding the following packages:

```bash
# PyTorch (CUDA version) - Index URL adjusted based on CUDA version
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# TensorFlow (GPU version)
uv add tensorflow[and-cuda]

# Other useful packages
uv add numpy pandas matplotlib scikit-learn jupyter

# CUDA-specific tools
uv add cupy-cuda124x  # CuPy for GPU-accelerated NumPy
```


### Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run hooks on all files
uv run pre-commit run --all-files
```

### Production Deployment

An optimized Dockerfile for production is also provided:

```bash
# Build production image
docker build -f Dockerfile.production -t bfrb:production .

# Run in production environment
docker run -p 8000:8000 bfrb:production
```

## Project Structure

```
bfrb/
├── .devcontainer/          # DevContainer configuration
│   ├── devcontainer.json  # Dev Container settings
│   └── Dockerfile         # Development Dockerfile
├── src/
│   └── bfrb/   # Main source code
├── tests/                  # Test files
├── pyproject.toml         # Project configuration and uv dependencies
├── uv.lock               # Dependency lock file
├── Dockerfile.production  # Production-optimized Dockerfile
├── .dockerignore          # Docker build context exclusions
└── README.md             # This file
```

## License
This project is licensed under the MIT License.
