# XFUND Generator

A Python toolkit for generating XFUND-style OCR datasets with document templates and automatic annotation.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![PyPI](https://img.shields.io/pypi/v/xfund-generator)

## Features

- **Template-Based Generation** - Convert DOCX templates to annotated OCR datasets
- **Multiple Annotation Formats** - XFUND, FUNSD, WildReceipt with unified API
- **Advanced Augmentations** - Realistic document variations with configurable difficulty
- **Type Safety** - Full Pydantic v2 integration for validation

## Quick Start

### Installation

```bash
# Using pip
pip install xfund-generator

# Or from source
git clone https://github.com/danghoangnhan/XFUND_generator.git
cd XFUND_generator
pip install -e .
```

**Requirements:** Python 3.9+, LibreOffice (for DOCX conversion)

### Basic Usage

```python
from xfund_generator import GeneratorConfig, XFUNDGenerator

config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/data.csv",
    output_dir="output"
)

generator = XFUNDGenerator(config)
result = generator.generate_dataset()
print(f"Generated {result.generated_entries} entries")
```

### Command Line

```bash
xfund-generator --config config/example_config.json
xfund-generator --validate-only
```

## Documentation

Full documentation is available on the [GitHub Wiki](https://github.com/danghoangnhan/XFUND_generator/wiki):

- [Installation Guide](https://github.com/danghoangnhan/XFUND_generator/wiki/Installation)
- [Getting Started](https://github.com/danghoangnhan/XFUND_generator/wiki/Getting-Started)
- [Configuration](https://github.com/danghoangnhan/XFUND_generator/wiki/Configuration)
- [Annotation Formats](https://github.com/danghoangnhan/XFUND_generator/wiki/Annotation-Formats)
- [API Reference](https://github.com/danghoangnhan/XFUND_generator/wiki/API-Reference)
- [Testing Guide](https://github.com/danghoangnhan/XFUND_generator/wiki/Testing)

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
make test

# Code quality
make lint
make format
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [GitHub Wiki](https://github.com/danghoangnhan/XFUND_generator/wiki)
- [PyPI Package](https://pypi.org/project/xfund-generator/)
- [Issue Tracker](https://github.com/danghoangnhan/XFUND_generator/issues)
- [XFUND Dataset](https://github.com/doc-analysis/XFUND)
