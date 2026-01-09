# Getting Started

This guide covers the basics of using XFUND Generator.

## Prerequisites

Ensure you have:
- Python 3.9+ installed
- LibreOffice installed
- XFUND Generator installed (see [[Installation]])

## Project Structure

```
your_project/
├── data/
│   ├── csv/              # CSV data files
│   │   └── data.csv
│   └── templates_docx/   # DOCX templates
│       └── template.docx
├── output/               # Generated datasets
├── config/
│   └── config.json       # Configuration file
└── fonts/                # Custom fonts (optional)
```

## Basic Usage

### Python API

```python
from xfund_generator import GeneratorConfig, XFUNDGenerator

# Create configuration
config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/data.csv",
    output_dir="output"
)

# Generate dataset
generator = XFUNDGenerator(config)
result = generator.generate_dataset()

print(f"Generated {result.generated_entries} entries")
print(f"Output directory: {result.output_path}")
```

### Command Line

```bash
# With default configuration
xfund-generator

# With custom configuration
xfund-generator --config config/my_config.json

# Validate only (no generation)
xfund-generator --config config/my_config.json --validate-only

# Show help
xfund-generator --help
```

## Configuration File

Create `config/config.json`:

```json
{
  "templates_dir": "data/templates_docx",
  "csv_path": "data/csv/data.csv",
  "output_dir": "output",
  "document_type": "medical",
  "enable_augmentations": true,
  "augmentation_difficulty": "medium",
  "image_dpi": 300
}
```

## Preparing Data

### CSV Data File

Create `data/csv/data.csv` with fields matching your template:

```csv
patient_name,doctor_name,date,diagnosis
John Smith,Dr. Johnson,2024-01-15,Healthy
Jane Doe,Dr. Williams,2024-01-16,Follow-up needed
```

### DOCX Template

Create templates with placeholder fields using `{field_name}` syntax:

```
Patient Name: {patient_name}
Doctor: {doctor_name}
Date: {date}
Diagnosis: {diagnosis}
```

## Output

Generated files in `output/`:

```
output/
├── images/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── annotations/
│   ├── 0001.json
│   ├── 0002.json
│   └── ...
└── debug/              # If debug mode enabled
    ├── 0001_overlay.png
    └── ...
```

### Annotation Format

```json
{
  "annotations": [
    {
      "id": 0,
      "text": "Patient Name:",
      "box": [50, 100, 150, 120],
      "label": "question",
      "words": [
        {"text": "Patient", "box": [50, 100, 100, 120]},
        {"text": "Name:", "box": [105, 100, 150, 120]}
      ]
    }
  ]
}
```

## Using Annotation Formats

### XFUND Format

```python
from xfund_generator.form.xfund import XFUNDDataset

dataset = XFUNDDataset(image_path="image.png")
dataset.add_annotation(...)
json_output = dataset.to_json()
```

### FUNSD Format

```python
from xfund_generator.form.funsd import FUNSDDataset

dataset = FUNSDDataset(image_path="image.png")
json_output = dataset.to_json()
```

### WildReceipt Format

```python
from xfund_generator.form.wildreceipt import WildReceiptDataset

dataset = WildReceiptDataset(image_path="image.png")
json_output = dataset.to_json()
```

## Enabling Augmentations

```python
config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/data.csv",
    output_dir="output",
    enable_augmentations=True,
    augmentation_difficulty="medium"  # easy, medium, hard, extreme
)
```

## Next Steps

- [[Configuration]] - All configuration options
- [[Annotation-Formats]] - Format specifications
- [[API-Reference]] - Full API documentation
