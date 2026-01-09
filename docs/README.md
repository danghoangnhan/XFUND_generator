# XFUND Generator

A powerful Python toolkit for generating XFUND-style OCR datasets with document templates, automatic annotation, and advanced augmentations. Enhanced with Pydantic for type safety and validation.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Pydantic](https://img.shields.io/badge/Pydantic-2.0%2B-red)

---

## Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Development](#-development)
- [OCR Evaluation](#-ocr-evaluation-tools)
- [Contributing](#-contributing)

---

## Features

- **Template-Based Generation**: Convert DOCX templates to annotated OCR datasets
- **Multiple Document Types**: Support for medical forms, invoices, contracts, and general documents
- **Multiple Annotation Formats**: XFUND, FUNSD, WildReceipt with unified API
- **Advanced Augmentations**: Realistic document variations with configurable difficulty
- **Type Safety**: Full Pydantic v2 integration for validation and error prevention
- **Quality Validation**: Automated quality checks and validation
- **OCR Evaluation**: Built-in tools for OCR model performance analysis

---

## Quick Start

### Installation

```bash
git clone https://github.com/danghoangnhan/XFUND_generator.git
cd XFUND_generator

# Install with uv (recommended)
make dev

# Or with pip
pip install -e ".[dev]"
```

### Basic Usage

```python
from xfund_generator import GeneratorConfig, XFUNDGenerator

# Create validated configuration
config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/data.csv",
    output_dir="output"
)

# Generate dataset
generator = XFUNDGenerator(config)
result = generator.generate_dataset()

print(f"Generated {result.generated_entries} entries")
```

### Command Line Usage

```bash
# Generate dataset with default config
xfund-generator

# Use custom configuration
xfund-generator --config config/example_config.json

# Validate setup only
xfund-generator --validate-only

# Show help
xfund-generator --help
```

---

## Project Structure

```
XFUND_generator/
├── xfund_generator/              # Core package
│   ├── __init__.py               # Package exports
│   ├── models.py                 # Pydantic models
│   ├── generate_dataset.py       # Main generator
│   ├── renderer.py               # Document rendering
│   ├── augmentations.py          # Image augmentations
│   ├── docx_utils.py             # DOCX processing
│   ├── utils.py                  # Utility functions
│   └── form/                     # Annotation formats
│       ├── base.py               # Base classes
│       ├── xfund.py              # XFUND format
│       ├── funsd.py              # FUNSD format
│       └── wildreceipt.py        # WildReceipt format
├── tests/                        # Test suite
├── config/                       # Configuration examples
├── docs/                         # Documentation
├── .github/workflows/            # CI/CD pipelines
├── pyproject.toml                # Project configuration
└── Makefile                      # Development commands
```

---

## Configuration

### Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `templates_dir` | string | `"data/templates_docx"` | Path to DOCX templates |
| `csv_path` | string | `"data/csv/data.csv"` | Path to CSV data file |
| `output_dir` | string | `"output"` | Output directory |
| `fonts_dir` | string | `"fonts/handwritten_fonts"` | Path to custom fonts |
| `document_type` | string | `"medical"` | Type: `medical`, `form`, `invoice`, `contract`, `general` |
| `enable_augmentations` | boolean | `true` | Enable image augmentations |
| `augmentation_difficulty` | string | `"medium"` | Level: `easy`, `medium`, `hard`, `extreme` |
| `image_dpi` | integer | `300` | Image resolution in DPI |
| `target_size` | integer | `1000` | Target image size in pixels |
| `add_bbox_jitter` | boolean | `true` | Add random jitter to bounding boxes |
| `strict_validation` | boolean | `true` | Enable strict validation mode |
| `generate_debug_overlays` | boolean | `true` | Generate debug overlay images |
| `max_workers` | integer | `4` | Maximum worker threads |
| `batch_size` | integer | `10` | Batch size for processing |

### Example Configuration File

```json
{
  "templates_dir": "data/templates_docx",
  "csv_path": "data/csv/data.csv",
  "output_dir": "output",
  "document_type": "medical",
  "enable_augmentations": true,
  "augmentation_difficulty": "medium",
  "image_dpi": 300,
  "target_size": 1000
}
```

### Creating Custom Configurations

```bash
cp config/example_config.json config/my_config.json
# Edit my_config.json
xfund-generator --config config/my_config.json
```

---

## Testing

### Quick Test Commands

```bash
# Run all tests
make test

# Run with coverage
make test-cov
```

### Using pytest Directly

```bash
# All tests
uv run pytest tests/ -v

# By marker
uv run pytest -m "unit" -v           # Unit tests only
uv run pytest -m "integration" -v    # Integration tests
uv run pytest -m "pydantic" -v       # Pydantic tests
uv run pytest -m "forms" -v          # Form class tests
uv run pytest -m "not slow" -v       # Exclude slow tests

# Specific files
uv run pytest tests/test_pydantic_models.py -v
uv run pytest tests/test_form_classes.py -v
uv run pytest tests/test_generator.py -v
uv run pytest tests/test_integration.py -v

# With coverage
uv run pytest --cov=xfund_generator --cov-report=html
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | Fast unit tests |
| `@pytest.mark.integration` | End-to-end tests |
| `@pytest.mark.pydantic` | Pydantic model tests |
| `@pytest.mark.forms` | Form class tests |
| `@pytest.mark.config` | Configuration tests |
| `@pytest.mark.slow` | Long-running tests |

### Test Coverage Areas

**Pydantic Models (`test_pydantic_models.py`):**
- BBoxModel validation and computed properties
- GeneratorConfig validation and path resolution
- DataRecord and TemplateValidationResult
- Model serialization/deserialization

**Form Classes (`test_form_classes.py`):**
- Base classes (Word, BaseAnnotation, BaseDataset)
- XFUND, FUNSD, WildReceipt format classes
- Unified JSON export API
- Template Method pattern implementation

**Integration (`test_integration.py`):**
- XFUND form generator functionality
- Question-answer linking automation
- Word-level annotation creation
- Complete pipeline testing

**Generator Core (`test_generator.py`):**
- DOCX processing utilities
- Word rendering functionality
- Image augmentation features

### Debugging Tests

```bash
# Run only failed tests
uv run pytest --lf -v

# Debug mode with detailed output
uv run pytest -v -s --tb=long --showlocals

# Stop on first failure
uv run pytest -x -v

# Pattern matching
uv run pytest -k "test_bbox" -v
```

---

## Development

### Make Commands

| Command | Description |
|---------|-------------|
| `make dev` | Install with dev dependencies |
| `make install` | Install production dependencies |
| `make test` | Run all tests |
| `make test-cov` | Run tests with coverage |
| `make lint` | Check code style (ruff + flake8) |
| `make format` | Format code with ruff |
| `make fix` | Auto-fix linting issues |
| `make type-check` | Run mypy type checking |
| `make build` | Build package |
| `make clean` | Remove build artifacts |

### Code Quality

```bash
# Check style
make lint

# Auto-fix issues
make fix

# Type checking
make type-check
```

### CI/CD

GitHub Actions runs automatically on:
- Push to `master`/`main`
- Pull requests
- Tag pushes (`v*`) for releases

**Tested Python versions:** 3.9, 3.10, 3.11, 3.12, 3.13, 3.14

---

## OCR Evaluation Tools

Comprehensive OCR model evaluation with precision, recall, and F1 metrics:

```bash
# Analyze OCR results
python analyze_ocr_results.py --excel-file ocr_results.xlsx

# Calculate metrics for 6 categories
python calculate_r1_precision_recall.py --excel-file ocr_results.xlsx

# Show summary
python show_ocr_summary.py
```

---

## Type Safety with Pydantic

```python
from xfund_generator import BBoxModel, DataRecord, XFUNDEntity

# Validated bounding box
bbox = BBoxModel(x1=10, y1=20, x2=100, y2=80)
print(f"Area: {bbox.area}")

# Validated data record
record = DataRecord(
    hospital_name_text="Central Hospital",
    doctor_name_text="Dr. Smith"
)

# Validated XFUND entity
entity = XFUNDEntity(
    id=0,
    text="Patient Name:",
    bbox=bbox,
    label="QUESTION"
)
```

---

## Output Format

The generator produces:

- **Images**: High-quality PNG/JPEG images from rendered documents
- **Annotations**: XFUND-format JSON annotations with entity information
- **Debug**: Optional debug overlays showing detected regions
- **Reports**: Generation statistics and quality metrics

### Example XFUND Annotation

```json
{
  "form": [
    {
      "id": 0,
      "text": "Patient Name:",
      "bbox": [50, 100, 150, 120],
      "label": "QUESTION",
      "words": ["Patient", "Name:"]
    },
    {
      "id": 1,
      "text": "John Smith",
      "bbox": [160, 100, 250, 120],
      "label": "ANSWER",
      "words": ["John", "Smith"]
    }
  ]
}
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Run quality checks (`make lint && make test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Adding New Tests

1. Choose appropriate test file based on functionality
2. Add relevant markers (`@pytest.mark.unit`, etc.)
3. Use existing fixtures from `conftest.py`
4. Follow naming: `test_feature_description`

---

## Requirements

- Python 3.9+
- LibreOffice (for DOCX to PDF conversion)

### Core Dependencies

- `pydantic>=2.0.0` - Type validation
- `pillow` - Image processing
- `opencv-python` - Computer vision
- `pandas` - Data handling
- `numpy` - Numerical operations

---

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

## Related Projects

- [XFUND Dataset](https://github.com/doc-analysis/XFUND) - Original XFUND dataset
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR toolkit
- [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3) - Document AI model

---

**Made with by [danghoangnhan](https://github.com/danghoangnhan)**
