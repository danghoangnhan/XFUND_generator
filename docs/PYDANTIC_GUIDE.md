# Pydantic Integration Guide

This guide shows how to use the new Pydantic support in the XFUND generator for better type safety, validation, and error handling.

## Overview

Pydantic has been added to provide:
- **Automatic data validation** with clear error messages
- **Type safety** with runtime validation
- **Better IDE support** with type hints and autocompletion
- **Easy JSON serialization/deserialization**
- **Structured configuration management**

## Key Components

### 1. Models (`src/models.py`)

The following Pydantic models are available:

#### BBoxModel
Validates bounding box coordinates:
```python
from src.models import BBoxModel

# Valid bbox
bbox = BBoxModel(x1=10, y1=20, x2=100, y2=80)
print(bbox.area())  # Calculate area
print(bbox.to_xfund_format())  # Convert to XFUND integer format
```

#### GeneratorConfig
Validates and manages configuration:
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

# Save to JSON
config.to_json_file("my_config.json")

# Load from JSON with validation
config = GeneratorConfig.from_json_file("my_config.json")
```

#### DataRecord
Validates CSV data records:
```python
from src.models import DataRecord

record = DataRecord(
    hospital_name_text="Central Hospital",
    doctor_name_text="Dr. Smith",
    diagnose_text="Hypertension"
)

# Access fields safely
hospital = record.get_field("hospital_name_text")
```

#### XFUNDEntity & XFUNDAnnotation
Validates XFUND annotations:
```python
from src.models import XFUNDEntity, XFUNDAnnotation, BBoxModel

bbox = BBoxModel(x1=10, y1=20, x2=100, y2=40)
entity = XFUNDEntity(
    id=0,
    text="Patient Name:",
    bbox=bbox,
    label="QUESTION"
)

annotation = XFUNDAnnotation(
    form=[entity],
    image_path="image.png"
)
```

### 2. Enhanced Utilities (`src/utils.py`)

New utility functions that work with Pydantic models:

- `load_csv_data_as_models()` - Load CSV data as validated DataRecord models
- `save_xfund_annotation_pydantic()` - Save validated XFUND annotations
- `load_config_with_validation()` - Load and validate configuration files
- `validate_annotation_quality()` - Quality validation with detailed feedback

### 3. Updated Generator (`src/generate_dataset.py`)

The main generator now supports:
- Validated configuration input
- Enhanced error reporting
- Type-safe data processing
- Automatic validation at each step

## Usage Examples

### Basic Usage

```python
from src.models import GeneratorConfig, get_default_config
from src.generate_dataset import XFUNDGenerator

# Create validated config
config = get_default_config()
config.output_dir = "my_output"
config.enable_augmentations = True

# Initialize generator
generator = XFUNDGenerator(config)

# Generate dataset with full validation
result = generator.generate_dataset()

print(f"Success: {result.success}")
print(f"Generated: {result.generated_entries}")
print(f"Success rate: {result.success_rate:.1f}%")
```

### Configuration Management

```python
from src.models import GeneratorConfig, validate_config_file

# Validate existing config
validation = validate_config_file("config.json")
if validation.is_valid:
    config = GeneratorConfig.from_json_file("config.json")
else:
    print("Validation errors:")
    for error in validation.errors:
        print(f"  - {error}")
```

### Working with Bounding Boxes

```python
from src.models import BBoxModel

# Create and validate bbox
bbox = BBoxModel(x1=10, y1=20, x2=100, y2=80)

# Normalize to XFUND scale
normalized = bbox.normalize(img_width=800, img_height=600, target_size=1000)

# Convert to different formats
xfund_coords = bbox.to_xfund_format()  # [10, 20, 100, 80]
float_coords = bbox.to_list()          # [10.0, 20.0, 100.0, 80.0]
```

### Error Handling

Pydantic provides detailed validation errors:

```python
from src.models import BBoxModel
from pydantic import ValidationError

try:
    invalid_bbox = BBoxModel(x1=100, y1=20, x2=10, y2=80)  # x1 >= x2
except ValidationError as e:
    print(f"Validation failed: {e}")
    # Shows exactly what went wrong with helpful messages
```

## Document Types

Supported document types:
- `DocumentType.MEDICAL` - Medical reports and forms
- `DocumentType.FORM` - General forms
- `DocumentType.INVOICE` - Invoices and receipts
- `DocumentType.CONTRACT` - Contracts and legal documents
- `DocumentType.GENERAL` - General documents

## Augmentation Levels

- `AugmentationDifficulty.EASY` - Light augmentations
- `AugmentationDifficulty.MEDIUM` - Standard augmentations
- `AugmentationDifficulty.HARD` - Strong augmentations
- `AugmentationDifficulty.EXTREME` - Very strong augmentations

## Command Line Usage

Create a default configuration:
```bash
python -c "from src.models import get_default_config; get_default_config().to_json_file('config.json')"
```

Run with validation:
```bash
python src/generate_dataset.py --config config_with_pydantic.json --validate-only
```

## Testing the Integration

Run the demonstration script:
```bash
python demo_pydantic_integration.py
```

This will test all the Pydantic models and show validation examples.

## Benefits

1. **Type Safety**: Catch errors at runtime with clear messages
2. **Better IDE Support**: Full autocomplete and type hints
3. **Validation**: Automatic validation of all input data
4. **Documentation**: Self-documenting code with clear field descriptions
5. **Serialization**: Easy conversion to/from JSON for config management
6. **Error Messages**: Clear, helpful validation error messages

## Migration Guide

### Old Way (Dict-based)
```python
config = {
    "templates_dir": "data/templates",
    "csv_path": "data.csv",
    "output_dir": "output"
}
generator = XFUNDGenerator(config)
```

### New Way (Pydantic)
```python
config = GeneratorConfig(
    templates_dir="data/templates",
    csv_path="data.csv", 
    output_dir="output"
)
generator = XFUNDGenerator(config)
```

The new way provides automatic validation, type checking, and better error messages!