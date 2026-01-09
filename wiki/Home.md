# XFUND Generator

A Python toolkit for generating XFUND-style OCR datasets with document templates and automatic annotation.

## Overview

XFUND Generator converts DOCX document templates into annotated OCR datasets compatible with XFUND, FUNSD, and WildReceipt formats. It supports advanced image augmentations and provides type-safe configuration through Pydantic v2.

## Key Features

| Feature | Description |
|---------|-------------|
| Template-Based Generation | Convert DOCX templates to annotated OCR datasets |
| Multiple Annotation Formats | XFUND, FUNSD, WildReceipt with unified API |
| Advanced Augmentations | Realistic document variations (blur, noise, rotation, perspective) |
| Type Safety | Full Pydantic v2 integration for validation |
| Quality Validation | Automated quality checks for generated annotations |
| OCR Evaluation | Built-in tools for OCR model performance analysis |

## Quick Navigation

### Getting Started
- [[Installation]] - System requirements and installation methods
- [[Getting-Started]] - First steps and basic usage

### Reference
- [[Configuration]] - All configuration options
- [[Annotation-Formats]] - XFUND, FUNSD, WildReceipt format specifications
- [[API-Reference]] - Python API documentation

### Development
- [[Testing]] - Running and writing tests
- [[Contributing]] - How to contribute

## Requirements

- **Python**: 3.9+
- **LibreOffice**: Required for DOCX to PDF conversion
- **Core Dependencies**: pydantic, pillow, opencv-python, pandas, numpy

## Installation

```bash
pip install xfund-generator
```

Or from source:

```bash
git clone https://github.com/danghoangnhan/XFUND_generator.git
cd XFUND_generator
pip install -e .
```

## Basic Usage

```python
from xfund_generator import GeneratorConfig, XFUNDGenerator

config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/data.csv",
    output_dir="output"
)

generator = XFUNDGenerator(config)
result = generator.generate_dataset()
```

## Links

- [GitHub Repository](https://github.com/danghoangnhan/XFUND_generator)
- [PyPI Package](https://pypi.org/project/xfund-generator/)
- [Issue Tracker](https://github.com/danghoangnhan/XFUND_generator/issues)
