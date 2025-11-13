# Installation Guide

This guide covers the installation process for the XFUND Generator on different platforms.

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for the project and dependencies
- **OS**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)

### Recommended Requirements
- **Python**: 3.10 or higher
- **RAM**: 16GB for large dataset generation
- **Storage**: 10GB+ for datasets and models
- **GPU**: CUDA-compatible GPU for accelerated augmentations (optional)

## Prerequisites

### LibreOffice Installation
LibreOffice is required for DOCX to PDF conversion.

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install libreoffice
```

#### CentOS/RHEL
```bash
sudo yum install libreoffice
```

#### macOS
```bash
# Using Homebrew
brew install --cask libreoffice

# Or download from https://www.libreoffice.org/download/
```

#### Windows
Download and install from [LibreOffice website](https://www.libreoffice.org/download/).

### Python Environment
Ensure Python 3.8+ is installed:
```bash
python --version  # Should show 3.8+
```

## Installation Methods

### Method 1: Git Clone (Recommended)

```bash
# Clone the repository
git clone https://github.com/danghoangnhan/XFUND_generator.git
cd XFUND_generator

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Download ZIP

1. Download ZIP from GitHub
2. Extract to desired directory
3. Open terminal in the extracted directory
4. Follow virtual environment and dependency installation steps above

### Method 3: Development Installation

For contributors and developers:

```bash
git clone https://github.com/danghoangnhan/XFUND_generator.git
cd XFUND_generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

## Dependency Installation

### Core Dependencies

The main dependencies are automatically installed with `pip install -r requirements.txt`:

- **pydantic>=2.0.0** - Data validation and settings management
- **pillow>=10.0.0** - Image processing
- **opencv-python>=4.8.0** - Computer vision operations
- **pandas>=2.0.0** - Data manipulation
- **numpy>=1.24.0** - Numerical computing
- **python-docx>=0.8.11** - DOCX file handling

### Optional Dependencies

For enhanced features, install optional dependencies:

```bash
# For BERT-based evaluation
pip install torch>=1.9.0 transformers>=4.21.0

# For semantic evaluation
pip install sentence-transformers>=2.2.0 bert-score>=0.3.13

# For advanced OCR evaluation
pip install scikit-learn>=1.0.0
```

### GPU Support (Optional)

For CUDA-accelerated operations:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Verification

### Basic Verification

Test the installation:

```bash
python -c "from src.models import GeneratorConfig; print('✅ Installation successful!')"
```

### Run Demo

Test with the demo script:

```bash
python demo_pydantic_integration.py
```

Expected output:
```
Pydantic Integration Demonstration for XFUND Generator
============================================================
=== BBox Model Validation Demo ===
✓ Valid bbox created: [10.0, 20.0, 100.0, 80.0]
...
```

### LibreOffice Verification

Test LibreOffice installation:

```bash
# Linux/macOS
libreoffice --version

# Windows (in Command Prompt)
"C:\Program Files\LibreOffice\program\soffice.exe" --version
```

## Configuration

### Initial Setup

1. **Create data directories** (if they don't exist):
```bash
mkdir -p data/csv data/templates_docx output
```

2. **Download sample data** (optional):
```bash
# Add your CSV data to data/csv/
# Add your DOCX templates to data/templates_docx/
```

3. **Test configuration**:
```bash
python -c "
from src.models import get_default_config
config = get_default_config()
print('Default config created successfully')
"
```

## Environment Variables (Optional)

Set environment variables for custom paths:

```bash
# Linux/macOS
export XFUND_DATA_DIR=/path/to/your/data
export XFUND_OUTPUT_DIR=/path/to/output

# Windows
set XFUND_DATA_DIR=C:\path\to\your\data
set XFUND_OUTPUT_DIR=C:\path\to\output
```

## Docker Installation (Alternative)

For containerized deployment:

```bash
# Build Docker image
docker build -t xfund-generator .

# Run container
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output xfund-generator
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'pydantic'
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/macOS
pip install pydantic
```

#### LibreOffice not found
```bash
# Check LibreOffice installation
which libreoffice  # Linux/macOS
where soffice.exe  # Windows

# Add to PATH if needed
export PATH=$PATH:/path/to/libreoffice/bin
```

#### Permission errors
```bash
# Fix permissions on Linux/macOS
chmod +x src/generate_dataset.py
sudo chown -R $USER:$USER /path/to/XFUND_generator
```

#### CUDA errors (GPU installation)
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall CPU version if GPU not needed
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Platform-Specific Issues

#### Windows

- Use PowerShell or Command Prompt as Administrator
- Ensure Python is in PATH
- Use forward slashes in paths or raw strings

#### macOS

- Install Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for package management
- May need to install certificates: `/Applications/Python\ 3.x/Install\ Certificates.command`

#### Linux

- Install development packages: `sudo apt-get install python3-dev build-essential`
- Some systems may need `python3-venv`: `sudo apt-get install python3-venv`

## Updating

To update to the latest version:

```bash
cd XFUND_generator
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstallation

To remove the installation:

```bash
# Deactivate virtual environment
deactivate

# Remove project directory
rm -rf /path/to/XFUND_generator

# Remove virtual environment if created separately
rm -rf /path/to/venv
```

## Next Steps

After successful installation:

1. Read the [Getting Started Guide](getting_started.md)
2. Review [Configuration Options](configuration.md)
3. Try [Basic Examples](examples/basic.md)

---

*Need help? Check the [Troubleshooting Guide](troubleshooting.md) or open an [issue](https://github.com/danghoangnhan/XFUND_generator/issues).*