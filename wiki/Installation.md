# Installation Guide

## System Requirements

### Minimum
- **Python**: 3.9+
- **RAM**: 4GB
- **Storage**: 2GB free space
- **OS**: Windows 10/11, macOS 10.14+, Linux (Ubuntu 18.04+)

### Recommended
- **Python**: 3.11+
- **RAM**: 16GB for large dataset generation
- **Storage**: 10GB+ for datasets
- **GPU**: CUDA-compatible (optional, for accelerated augmentations)

## Prerequisites

### LibreOffice

LibreOffice is required for DOCX to PDF conversion.

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libreoffice
```

**CentOS/RHEL:**
```bash
sudo yum install libreoffice
```

**macOS:**
```bash
brew install --cask libreoffice
```

**Windows:**
Download from [LibreOffice website](https://www.libreoffice.org/download/).

## Installation Methods

### Method 1: pip (Recommended)

```bash
pip install xfund-generator
```

### Method 2: From Source

```bash
git clone https://github.com/danghoangnhan/XFUND_generator.git
cd XFUND_generator
pip install -e .
```

### Method 3: Development Installation

```bash
git clone https://github.com/danghoangnhan/XFUND_generator.git
cd XFUND_generator
pip install -e ".[dev]"
```

### Method 4: Using uv

```bash
git clone https://github.com/danghoangnhan/XFUND_generator.git
cd XFUND_generator
make dev
```

## Dependencies

### Core Dependencies

Automatically installed:
- `pydantic>=2.0.0` - Data validation
- `pillow` - Image processing
- `opencv-python` - Computer vision
- `pandas` - Data handling
- `numpy` - Numerical operations
- `python-docx` - DOCX processing
- `pdf2image` - PDF to image conversion
- `albumentations` - Image augmentations

### Optional Dependencies

```bash
# BERT-based evaluation
pip install torch transformers

# Semantic evaluation
pip install sentence-transformers bert-score
```

### GPU Support

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Verification

### Test Installation

```bash
python -c "from xfund_generator import GeneratorConfig; print('Installation successful')"
```

### Verify LibreOffice

```bash
libreoffice --version
```

## Initial Setup

```bash
# Create data directories
mkdir -p data/csv data/templates_docx output

# Test with example config
xfund-generator --config config/example_config.json --validate-only
```

## Troubleshooting

### ImportError: No module named 'pydantic'
```bash
pip install pydantic>=2.0.0
```

### LibreOffice not found
```bash
# Check installation
which libreoffice

# Add to PATH if needed
export PATH=$PATH:/path/to/libreoffice/bin
```

### Permission errors (Linux/macOS)
```bash
chmod +x xfund_generator/generate_dataset.py
```

## Updating

```bash
pip install --upgrade xfund-generator
```

Or from source:
```bash
git pull origin main
pip install -e . --upgrade
```

## Next Steps

- [[Getting-Started]] - Basic usage guide
- [[Configuration]] - Configuration options
