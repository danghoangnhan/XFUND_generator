# XFUND Generator

A powerful Python toolkit for generating XFUND-style OCR datasets with document templates, automatic annotation, and advanced augmentations. Enhanced with Pydantic for type safety and validation.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Pydantic](https://img.shields.io/badge/Pydantic-2.0%2B-red)

## 🌟 Features

- **Template-Based Generation**: Convert DOCX templates to annotated OCR datasets
- **Multiple Document Types**: Support for medical forms, invoices, contracts, and general documents
- **Advanced Augmentations**: Realistic document variations with configurable difficulty
- **Type Safety**: Full Pydantic integration for validation and error prevention
- **XFUND Format**: Generates annotations in XFUND standard format
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

### Command Line Usage

```bash
# Generate dataset with default config
python src/generate_dataset.py

# Use custom configuration
python src/generate_dataset.py --config medical_config.json

# Validate setup only
python src/generate_dataset.py --config config.json --validate-only
```

## 📁 Project Structure

```
XFUND_generator/
├── src/                          # Core source code
│   ├── models.py                 # Pydantic models for validation
│   ├── generate_dataset.py       # Main dataset generation
│   ├── utils.py                  # Utility functions
│   ├── renderer.py               # Document rendering
│   ├── augmentations.py          # Image augmentations
│   └── docx_utils.py             # DOCX processing
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

The project uses Pydantic for type-safe configuration management:

### Basic Configuration

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

## 📊 OCR Evaluation Tools

Comprehensive OCR model evaluation with precision, recall, and F1 metrics:

```bash
# Analyze OCR results
python analyze_ocr_results.py --excel-file ocr_results.xlsx

# Calculate R1, precision, recall for 6 categories
python calculate_r1_precision_recall.py --excel-file ocr_results.xlsx

# Show clean summary
python show_ocr_summary.py
```

## 🛡️ Type Safety with Pydantic

The project features comprehensive Pydantic integration:

```python
from src.models import BBoxModel, DataRecord, XFUNDEntity

# Validated bounding box
bbox = BBoxModel(x1=10, y1=20, x2=100, y2=80)
print(f"Area: {bbox.area()}")

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

## 📚 Examples & Demos

Run the included demo scripts to see the features in action:

```bash
# Basic Pydantic functionality
python demo_pydantic_integration.py

# Configuration examples
python pydantic_examples.py

# Complete feature overview
python pydantic_summary.py
```

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Test specific component
python tests/test_generator.py

# Validate configuration
python src/generate_dataset.py --config config.json --validate-only
```

## 📖 Documentation

- [Pydantic Integration Guide](PYDANTIC_GUIDE.md) - Complete guide to using Pydantic features
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Technical implementation details
- [Semantic Evaluation](SEMANTIC_EVALUATION_README.md) - Semantic evaluation tools
- [Configuration Guide](CONFIG_GUIDE.md) - Configuration options and examples

## 🔄 Workflow

1. **Prepare Templates**: Create DOCX templates with layout definitions
2. **Prepare Data**: CSV files with field data to populate templates
3. **Configure**: Set up configuration with desired options
4. **Generate**: Run the generator to create annotated dataset
5. **Evaluate**: Use built-in tools to assess OCR performance

## 🎯 Output Format

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Requirements

- Python 3.8+
- LibreOffice (for DOCX to PDF conversion)
- Required Python packages (see `requirements.txt`)

### Core Dependencies

- `pydantic>=2.0.0` - Type validation and serialization
- `pillow>=10.0.0` - Image processing
- `opencv-python>=4.8.0` - Computer vision operations
- `pandas>=2.0.0` - Data handling
- `numpy>=1.24.0` - Numerical operations

### Optional Dependencies

- `torch>=1.9.0` - For BERT-based evaluation
- `transformers>=4.21.0` - Language models
- `sentence-transformers>=2.2.0` - Semantic evaluation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [XFUND Dataset](https://github.com/doc-analysis/XFUND) - Original XFUND dataset
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR toolkit
- [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3) - Document AI model

## 📞 Support

- 📫 **Issues**: [GitHub Issues](https://github.com/danghoangnhan/XFUND_generator/issues)
- 📚 **Documentation**: See `docs/` directory
- 💬 **Discussions**: [GitHub Discussions](https://github.com/danghoangnhan/XFUND_generator/discussions)

## 🙏 Acknowledgments

- XFUND dataset creators for the annotation format
- PaddleOCR team for OCR evaluation methods
- Pydantic team for the excellent validation library

---

**Made with ❤️ by [danghoangnhan](https://github.com/danghoangnhan)**