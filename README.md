# XFUND Generator

A  toolkit for generating XFUND-style OCR datasets with document templates, automatic annotation.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Pydantic](https://img.shields.io/badge/Pydantic-2.0%2B-red)

## 🌟 Features

- **Template-Based Generation**: Convert DOCX templates to annotated OCR datasets
- **Multiple Annotation Formats**: Support for XFUND, FUNSD, and WildReceipt formats with unified API
- **Form Classes with OOP Inheritance**: Extensible architecture for adding new annotation formats
- **Advanced Augmentations**: Realistic document variations with configurable difficulty
- **Type Safety**: Full Pydantic v2 integration for validation and error prevention
- **Unified JSON Export**: Single `to_json()` method works across all annotation formats
- **Quality Validation**: Automated quality checks and validation
- **OCR Evaluation**: Built-in tools for OCR model performance analysis

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/danghoangnhan/XFUND_generator.git
cd XFUND_generator

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.models import GeneratorConfig
from src.generate_dataset import XFUNDGenerator

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

### Form Classes with Unified API

```python
from src.form.xfund import XFUNDDataset
from src.form.funsd import FUNSDDataset
from src.form.wildreceipt import WildReceiptDataset

# All formats use the same unified API
xfund_dataset = XFUNDDataset(image_path="image.png")
funsd_dataset = FUNSDDataset(image_path="image.png")
wild_dataset = WildReceiptDataset(image_path="image.png")

# Consistent JSON export across all formats
xfund_json = xfund_dataset.to_json()    # XFUND format with linking
funsd_json = funsd_dataset.to_json()    # FUNSD format with key/value IDs
wild_json = wild_dataset.to_json()      # WildReceipt format (minimal)

# Polymorphic usage - works with any format
def export_dataset(dataset):
    return dataset.to_json()  # Works for all formats!
```

### Command Line Usage

```bash
# Generate dataset with default config
python src/generate_dataset.py

# Use example configuration
python src/generate_dataset.py --config config/example_config.json

# Validate setup only
python src/generate_dataset.py --config config/example_config.json --validate-only
```

## 📁 Project Structure

```
XFUND_generator/
├── src/                          # Core source code
│   ├── models.py                 # Pydantic v2 models for validation
│   ├── generate_dataset.py       # Main dataset generation
│   ├── form/                     # Form classes with OOP inheritance
│   │   ├── base.py              # BaseDataset with unified to_json()
│   │   ├── xfund.py             # XFUND format implementation
│   │   ├── funsd.py             # FUNSD format implementation
│   │   └── wildreceipt.py       # WildReceipt format implementation
│   ├── xfund_form_integration.py # XFUND form generator integration
│   ├── utils.py                  # Utility functions
│   ├── renderer.py               # Document rendering
│   ├── augmentations.py          # Image augmentations
│   └── docx_utils.py             # DOCX processing with Pydantic
├── config/                       # Configuration files
│   └── example_config.json       # Example configuration
├── data/                         # Data files
│   ├── csv/                      # CSV data files
│   └── templates_docx/           # Document templates
├── docs/                         # Documentation
├── evaluation_results/           # OCR evaluation outputs
├── output/                       # Generated datasets
├── tests/                        # Test files
└── *.py                         # Demo and utility scripts
```

## 🔧 Configuration

The project uses Pydantic for type-safe configuration management. 

### Example Configuration

See `config/example_config.json` for a complete configuration example:

```json
{
  "templates_dir": "data/templates_docx",
  "csv_path": "data/csv/data.csv",
  "output_dir": "output",
  "fonts_dir": "fonts/handwritten_fonts",
  "document_type": "medical",
  "enable_augmentations": true,
  "augmentation_difficulty": "medium",
  "image_dpi": 300,
  "target_size": 1000,
  "add_bbox_jitter": true,
  "strict_validation": true,
  "strict_augmentation": false,
  "generate_debug_overlays": true,
  "max_workers": 4,
  "batch_size": 10
}
```

### Available Document Types

- `medical` - Medical reports and forms
- `form` - General forms and applications
- `invoice` - Invoices and receipts  
- `contract` - Contracts and legal documents
- `general` - General documents

### Augmentation Levels

- `easy` - Light augmentations (brightness, contrast)
- `medium` - Standard augmentations (blur, noise, rotation)
- `hard` - Strong augmentations (perspective, distortion)
- `extreme` - Very aggressive augmentations

## 🛡️ Type Safety with Pydantic v2

The project features comprehensive Pydantic v2 integration:

```python
from src.models import BBoxModel, DataRecord, XFUNDEntity, TemplateValidationResult

# Validated bounding box with computed properties  
bbox = BBoxModel(x1=10, y1=20, x2=100, y2=80)
print(f"Area: {bbox.area()}")          # Computed property
print(f"Center: {bbox.center()}")      # Tuple of center coordinates

# Validated data record with type checking
record = DataRecord(
    hospital_name_text="Central Hospital",
    doctor_name_text="Dr. Smith"
)

# Validated form annotation with automatic validation
from src.form.base import Word
from src.form.xfund import XFUNDAnnotation

annotation = XFUNDAnnotation(
    id=0,
    text="Patient Name:",
    box=[10, 20, 100, 80],
    label="question",
    words=[Word(text="Patient", box=[10, 20, 60, 80]), Word(text="Name:", box=[65, 20, 100, 80])],
    linking=[[0, 1]]
)

# Template validation with success/error handling
validation_result = TemplateValidationResult.create_success(
    template_path="template.docx",
    message="Template validated successfully"
)
```

## 🧪 Testing

The XFUND Generator includes a comprehensive test suite with **multiple ways to run tests**:

### 🥇 Recommended: Self-Contained Script (Modern Python)
```bash
# Uses inline script dependencies - most reliable!
uv run run_all_tests.py                  # Run all tests
uv run run_all_tests.py --quick          # Quick tests only  
uv run run_all_tests.py --coverage       # With coverage report
uv run run_all_tests.py --pydantic       # Pydantic model tests
uv run run_all_tests.py --help           # Show all options
```

### ⚡ Quick Development: Bash Script
```bash
./test.sh                                # All tests
./test.sh quick                          # Quick tests (fast development)
./test.sh pydantic                       # Pydantic tests only
./test.sh coverage                       # With coverage
./test.sh help                           # Show options
```

### 🏭 CI/CD: Make Commands  
```bash
make test                                # All tests
make test-quick                          # Quick tests
make test-coverage                       # With coverage report
make help                                # Show all make targets
```

### 🔧 Advanced: Direct pytest
```bash
python -m pytest tests/ -v              # All tests
python -m pytest -m "not slow" -v       # Quick tests
python -m pytest tests/test_pydantic_models.py  # Specific file
python -m pytest --cov=src tests/       # With coverage
```

### 📋 Quick Reference
```bash
# See all available test commands
./list_test_commands.sh

# Comprehensive documentation  
cat COMPLETE_TEST_GUIDE.md
```

**Test Categories**: `unit`, `integration`, `pydantic`, `forms`, `config`, `slow`

## 📖 Documentation

- [Pydantic Integration Guide](docs/PYDANTIC_GUIDE.md) - Complete guide to using Pydantic v2 features
- [Form Refactoring Summary](docs/FORM_REFACTORING_SUMMARY.md) - OOP inheritance architecture details
- [Why Unified JSON Methods](docs/WHY_UNIFIED_JSON_METHODS.md) - Benefits of eliminating code duplication
- [XFUND Form Integration](docs/XFUND_FORM_INTEGRATION_SUMMARY.md) - Form classes integration guide
- [Pydantic Improvements](docs/PYDANTIC_IMPROVEMENT_SUMMARY.md) - Migration from Dict[str, Any] to Pydantic
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Technical implementation details
- [Semantic Evaluation](SEMANTIC_EVALUATION_README.md) - Semantic evaluation tools
- [Configuration Guide](CONFIG_GUIDE.md) - Configuration options and examples

## 🔄 Workflow

1. **Prepare Templates**: Create DOCX templates with layout definitions
2. **Prepare Data**: CSV files with field data to populate templates
3. **Configure**: Set up configuration with desired options
4. **Generate**: Run the generator to create annotated dataset
5. **Evaluate**: Use built-in tools to assess OCR performance

## � Recent Improvements (November 2024)

### 1. Pydantic v2 Integration
- **Type Safety**: Replaced `Dict[str, Any]` return types with validated Pydantic models
- **Modern Validators**: Updated to Pydantic v2 syntax with `@model_validator` and `@field_validator`
- **Better Error Messages**: Comprehensive validation error reporting
- **Performance**: Improved validation speed and memory usage


## �🎯 Output Format

The generator produces:

- **Images**: High-quality PNG/JPEG images from rendered documents
- **Annotations**: Multi-format JSON annotations (XFUND/FUNSD/WildReceipt) with unified API
- **Debug**: Optional debug overlays showing detected regions
- **Reports**: Generation statistics and quality metrics

### Example Annotations (Unified API)

All formats use the same `to_json()` method but produce format-specific output:

**XFUND Format** (with linking information):
```json
{
  "annotations": [
    {
      "id": 0,
      "text": "Patient Name:",
      "box": [50, 100, 150, 120],
      "label": "question",
      "words": [{"text": "Patient", "box": [50, 100, 100, 120]}, {"text": "Name:", "box": [105, 100, 150, 120]}],
      "linking": [[0, 1]]
    },
    {
      "id": 1,
      "text": "John Smith",
      "box": [160, 100, 250, 120], 
      "label": "answer",
      "words": [{"text": "John", "box": [160, 100, 190, 120]}, {"text": "Smith", "box": [195, 100, 250, 120]}],
      "linking": []
    }
  ]
}
```

**FUNSD Format** (with key/value relationships):
```json
{
  "annotations": [
    {
      "id": 0,
      "text": "Patient Name:",
      "box": [50, 100, 150, 120],
      "label": "question",
      "words": [{"text": "Patient", "box": [50, 100, 100, 120]}, {"text": "Name:", "box": [105, 100, 150, 120]}],
      "key_id": 0,
      "value_id": 1
    }
  ]
}
```

**WildReceipt Format** (minimal structure):
```json
{
  "annotations": [
    {
      "id": 0,
      "text": "Patient Name:",
      "box": [50, 100, 150, 120],
      "label": "question",
      "words": [{"text": "Patient", "box": [50, 100, 100, 120]}, {"text": "Name:", "box": [105, 100, 150, 120]}]
    }
  ]
}
```

## ⚙️ Advanced Features

### Custom Augmentations

```python
from src.augmentations import DocumentAugmenter

augmenter = DocumentAugmenter(
    brightness_range=(0.7, 1.3),
    blur_probability=0.3,
    noise_probability=0.2
)
```

### Quality Validation

```python
from src.utils import validate_annotation_quality

issues = validate_annotation_quality(annotation)
if issues:
    print("Quality issues found:", issues)
```

### Batch Processing

```python
config = GeneratorConfig(
    # ... other settings
    max_workers=8,
    batch_size=20
)
```

## 📋 Requirements

- Python 3.8+
- LibreOffice (for DOCX to PDF conversion)
- Required Python packages (see `requirements.txt`)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [XFUND Dataset](https://github.com/doc-analysis/XFUND) - Original XFUND dataset
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR toolkit
- [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3) - Document AI model