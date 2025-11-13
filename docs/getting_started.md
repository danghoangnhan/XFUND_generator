# Getting Started Guide

This guide will help you get up and running with the XFUND Generator quickly.

## Overview

The XFUND Generator creates annotated OCR datasets from document templates. The typical workflow is:

1. **Prepare Templates** → 2. **Prepare Data** → 3. **Configure** → 4. **Generate** → 5. **Evaluate**

## Quick Start (5 Minutes)

### Step 1: Verify Installation

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows

# Test installation
python demo_pydantic_integration.py
```

### Step 2: Create Your First Dataset

```python
from src.models import GeneratorConfig
from src.generate_dataset import XFUNDGenerator

# Create configuration
config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/data.csv",
    output_dir="output/my_first_dataset"
)

# Generate dataset
generator = XFUNDGenerator(config)
result = generator.generate_dataset()

print(f"Success: {result.success}")
print(f"Generated: {result.generated_entries} entries")
```

### Step 3: Check Results

```bash
ls output/my_first_dataset/
# Should show:
# images/      - Generated document images
# annotations/ - XFUND format annotations
```

## Understanding the Components

### 1. Templates

Document templates define the layout and structure:

```
data/templates_docx/
├── medical_report.docx      # DOCX template file
└── medical_report_layout.json  # Layout definition
```

**Example layout.json:**
```json
{
  "hospital_name": [100, 50, 400, 80],
  "doctor_name": [100, 120, 300, 140],
  "diagnosis": [100, 200, 500, 250]
}
```

### 2. Data Files

CSV files contain the data to populate templates:

```csv
hospital_name_text,doctor_name_text,diagnose_text
Central Hospital,Dr. Smith,Hypertension
City Clinic,Dr. Johnson,Diabetes
```

### 3. Configuration

Pydantic models ensure type safety and validation:

```python
from src.models import GeneratorConfig, DocumentType

config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/data.csv", 
    output_dir="output",
    document_type=DocumentType.MEDICAL,
    enable_augmentations=True,
    image_dpi=300
)
```

## Basic Examples

### Example 1: Medical Forms

```python
from src.models import GeneratorConfig, DocumentType, AugmentationDifficulty

# Medical document configuration
config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/medical_data.csv",
    output_dir="output/medical_forms",
    document_type=DocumentType.MEDICAL,
    enable_augmentations=True,
    augmentation_difficulty=AugmentationDifficulty.MEDIUM,
    image_dpi=300,
    strict_validation=True
)

generator = XFUNDGenerator(config)
result = generator.generate_dataset()

print(f"Medical forms generated: {result.generated_entries}")
print(f"Success rate: {result.success_rate:.1f}%")
```

### Example 2: High-Quality Forms

```python
# High-quality form configuration (no augmentations)
config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/form_data.csv",
    output_dir="output/clean_forms",
    document_type=DocumentType.FORM,
    enable_augmentations=False,  # Clean output
    image_dpi=600,  # High resolution
    add_bbox_jitter=False  # Precise positioning
)

generator = XFUNDGenerator(config)
result = generator.generate_dataset()
```

### Example 3: Command Line Usage

```bash
# Generate with default settings
python src/generate_dataset.py

# Use custom configuration
python src/generate_dataset.py --config medical_config.json

# Validate configuration only
python src/generate_dataset.py --config config.json --validate-only

# Enable debug mode
python src/generate_dataset.py --config config.json --debug
```

## Working with Templates

### Creating a Template

1. **Design the DOCX template:**
   - Use placeholder text: `{{hospital_name_text}}`
   - Create consistent layout
   - Save as `.docx` format

2. **Define the layout JSON:**
   ```json
   {
     "hospital_name_text": [x1, y1, x2, y2],
     "doctor_name_text": [x1, y1, x2, y2]
   }
   ```

3. **Place files in templates directory:**
   ```
   data/templates_docx/
   ├── my_template.docx
   └── my_template_layout.json
   ```

### Template Best Practices

- Use consistent field naming between DOCX and CSV
- Ensure adequate spacing between fields
- Test with sample data before large-scale generation
- Consider different text lengths in your layout

## Working with Data

### CSV Format

Your CSV should match the template fields:

```csv
hospital_name_text,doctor_name_text,diagnose_text,doctor_comment_text
"Central Medical Center","Dr. Sarah Johnson","Hypertension","Patient shows improvement"
"City General Hospital","Dr. Michael Chen","Diabetes Type 2","Continue current medication"
```

### Data Validation

The system automatically validates data:

```python
from src.models import DataRecord

# This will validate the data
record = DataRecord(
    hospital_name_text="Central Hospital",
    doctor_name_text="Dr. Smith"
)

# Access validated data
hospital = record.get_field("hospital_name_text")
```

## Understanding Output

### Generated Files

```
output/
├── images/              # Generated document images
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── annotations/         # XFUND format annotations
│   ├── 0001.json
│   ├── 0002.json
│   └── ...
└── generation_report.json  # Generation statistics
```

### XFUND Annotation Format

```json
{
  "form": [
    {
      "id": 0,
      "text": "Central Hospital",
      "bbox": [100, 50, 400, 80],
      "label": "HEADER",
      "words": ["Central", "Hospital"]
    },
    {
      "id": 1,
      "text": "Patient Name:",
      "bbox": [100, 120, 200, 140],
      "label": "QUESTION"
    }
  ]
}
```

## Configuration Examples

### Minimal Configuration

```python
from src.models import GeneratorConfig

config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/data.csv",
    output_dir="output"
)
```

### Full Configuration

```python
from src.models import GeneratorConfig, DocumentType, AugmentationDifficulty

config = GeneratorConfig(
    # Required paths
    templates_dir="data/templates_docx",
    csv_path="data/csv/medical_data.csv",
    output_dir="output/medical_dataset",
    
    # Optional paths
    fonts_dir="fonts/handwritten_fonts",
    
    # Document settings
    document_type=DocumentType.MEDICAL,
    image_dpi=300,
    target_size=1000,
    
    # Augmentation settings
    enable_augmentations=True,
    augmentation_difficulty=AugmentationDifficulty.MEDIUM,
    
    # Quality settings
    strict_validation=True,
    add_bbox_jitter=True,
    generate_debug_overlays=True,
    
    # Performance settings
    max_workers=4,
    batch_size=10
)
```

## Common Workflows

### Workflow 1: Medical Documents

1. Create medical form template (DOCX + layout JSON)
2. Prepare medical data CSV
3. Configure for medical document type
4. Generate with medium augmentations
5. Evaluate OCR performance

### Workflow 2: Clean Forms

1. Create form template
2. Prepare form data
3. Configure with no augmentations, high DPI
4. Generate clean dataset
5. Use for OCR model training

### Workflow 3: Augmented Dataset

1. Create base template
2. Prepare diverse data
3. Configure with strong augmentations
4. Generate large dataset
5. Use for robustness testing

## Error Handling

### Common Validation Errors

```python
from pydantic import ValidationError

try:
    config = GeneratorConfig(
        templates_dir="nonexistent_dir",
        csv_path="missing.csv",
        output_dir="output"
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
    # Shows exactly what went wrong
```

### Generation Errors

```python
generator = XFUNDGenerator(config)
result = generator.generate_dataset()

if not result.success:
    print("Generation failed!")
    for error in result.errors:
        print(f"Error: {error}")
```

## Quality Validation

### Built-in Validation

```python
from src.utils import validate_annotation_quality

# This happens automatically with strict_validation=True
issues = validate_annotation_quality(annotation)
if issues:
    print("Quality issues found:", issues)
```

### Manual Validation

```bash
# Validate configuration
python src/generate_dataset.py --config config.json --validate-only

# Check generation results
python -c "
from src.models import GeneratorConfig
config = GeneratorConfig.from_json_file('medical_config.json')
print(f'Config valid: {config}')
"
```

## Next Steps

After completing this guide:

1. **Learn Advanced Features**: Read [Pydantic Integration](pydantic_integration.md)
2. **Explore Configuration**: See [Configuration Guide](configuration.md)
3. **Try Examples**: Check [Examples Directory](examples/)
4. **Evaluate OCR**: Learn about [OCR Evaluation](ocr_evaluation.md)

## Tips for Success

1. **Start Small**: Begin with a few templates and small datasets
2. **Validate Early**: Use `--validate-only` to catch issues early
3. **Check Output**: Review generated images and annotations
4. **Iterate**: Adjust configuration based on results
5. **Monitor Quality**: Use built-in validation tools

## Getting Help

- **Documentation**: Browse the [docs directory](.)
- **Examples**: Check example scripts in the project root
- **Issues**: Report problems on [GitHub Issues](https://github.com/danghoangnhan/XFUND_generator/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/danghoangnhan/XFUND_generator/discussions)

---

*Ready to dive deeper? Check out the [Advanced Examples](examples/advanced.md) or [Configuration Guide](configuration.md).*