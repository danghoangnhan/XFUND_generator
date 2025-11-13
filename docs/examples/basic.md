# Basic Examples

This guide provides simple, practical examples to get you started with the XFUND Generator.

## Example 1: Simple Medical Form

### Step 1: Prepare Data

Create a CSV file with medical data:

**data/csv/medical_simple.csv**
```csv
hospital_name_text,doctor_name_text,diagnose_text
Central Hospital,Dr. Smith,Hypertension
City Medical,Dr. Johnson,Diabetes
General Clinic,Dr. Brown,Flu
```

### Step 2: Create Configuration

```python
from src.models import GeneratorConfig, DocumentType

config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/medical_simple.csv",
    output_dir="output/simple_medical",
    document_type=DocumentType.MEDICAL,
    image_dpi=300,
    enable_augmentations=False  # Start simple
)

# Save configuration
config.to_json_file("simple_medical_config.json")
```

### Step 3: Generate Dataset

```python
from src.generate_dataset import XFUNDGenerator

generator = XFUNDGenerator(config)
result = generator.generate_dataset()

print(f"Generated: {result.generated_entries} entries")
print(f"Failed: {result.failed_entries} entries")
print(f"Success rate: {result.success_rate:.1f}%")
```

### Expected Output

```
Generated: 3 entries
Failed: 0 entries
Success rate: 100.0%
```

## Example 2: Form with Augmentations

### Configuration with Augmentations

```python
from src.models import GeneratorConfig, AugmentationDifficulty

config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/medical_simple.csv",
    output_dir="output/augmented_medical",
    document_type=DocumentType.MEDICAL,
    enable_augmentations=True,
    augmentation_difficulty=AugmentationDifficulty.MEDIUM,
    image_dpi=300
)

generator = XFUNDGenerator(config)
result = generator.generate_dataset()

print("Augmented dataset generated!")
print(f"Output directory: {result.output_paths['images']}")
```

## Example 3: High-Quality Dataset

### Configuration for High Quality

```python
config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/medical_simple.csv",
    output_dir="output/high_quality",
    document_type=DocumentType.MEDICAL,
    
    # High quality settings
    image_dpi=600,  # Higher resolution
    enable_augmentations=False,  # Clean images
    add_bbox_jitter=False,  # Precise positioning
    strict_validation=True,  # Strict quality checks
    generate_debug_overlays=True  # Show bounding boxes
)

generator = XFUNDGenerator(config)
result = generator.generate_dataset()
```

## Example 4: Batch Processing

### Process Multiple CSV Files

```python
import os
from pathlib import Path

# List of data files
csv_files = [
    "data/csv/medical_batch1.csv",
    "data/csv/medical_batch2.csv",
    "data/csv/medical_batch3.csv"
]

for i, csv_file in enumerate(csv_files):
    print(f"Processing batch {i+1}: {csv_file}")
    
    config = GeneratorConfig(
        templates_dir="data/templates_docx",
        csv_path=csv_file,
        output_dir=f"output/batch_{i+1}",
        document_type=DocumentType.MEDICAL
    )
    
    generator = XFUNDGenerator(config)
    result = generator.generate_dataset()
    
    print(f"  Generated: {result.generated_entries} entries")
```

## Example 5: Command Line Usage

### Basic Command Line

```bash
# Generate with default configuration
python src/generate_dataset.py \
  --templates-dir data/templates_docx \
  --csv-path data/csv/medical_simple.csv \
  --output-dir output/cli_generated
```

### Using Configuration File

```bash
# Save configuration first
python -c "
from src.models import get_default_config
config = get_default_config()
config.csv_path = 'data/csv/medical_simple.csv'
config.output_dir = 'output/from_config'
config.to_json_file('example_config.json')
"

# Generate using config file
python src/generate_dataset.py --config example_config.json
```

### Validation Only

```bash
# Validate configuration without generating
python src/generate_dataset.py \
  --config example_config.json \
  --validate-only
```

## Example 6: Working with Bounding Boxes

### Create and Validate Bounding Boxes

```python
from src.models import BBoxModel

# Create bounding box
bbox = BBoxModel(x1=50, y1=100, x2=300, y2=130)

print(f"Bounding box: {bbox.to_list()}")
print(f"Area: {bbox.area()}")
print(f"XFUND format: {bbox.to_xfund_format()}")

# Normalize for different image sizes
normalized = bbox.normalize(img_width=800, img_height=600, target_size=1000)
print(f"Normalized: {normalized.to_xfund_format()}")
```

### Output
```
Bounding box: [50.0, 100.0, 300.0, 130.0]
Area: 7500.0
XFUND format: [50, 100, 300, 130]
Normalized: [62, 166, 375, 216]
```

## Example 7: Data Record Validation

### Validate CSV Data

```python
from src.models import DataRecord

# Valid record
record = DataRecord(
    hospital_name_text="Central Medical Center",
    doctor_name_text="Dr. Sarah Johnson",
    diagnose_text="Hypertension, controlled"
)

print(f"Hospital: {record.hospital_name_text}")
print(f"Doctor: {record.doctor_name_text}")

# Access additional fields safely
patient_age = record.get_field("patient_age")  # Returns "" if not found
print(f"Patient age: '{patient_age}'")
```

### Handle Validation Errors

```python
from pydantic import ValidationError

try:
    # This will fail validation
    invalid_record = DataRecord(
        hospital_name_text="",  # Empty text
        doctor_name_text="Dr. Smith"
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
```

## Example 8: XFUND Annotation Creation

### Create Annotation Manually

```python
from src.models import XFUNDEntity, XFUNDAnnotation, BBoxModel

# Create entities
entities = []

# Hospital name entity
bbox1 = BBoxModel(x1=100, y1=50, x2=400, y2=80)
entity1 = XFUNDEntity(
    id=0,
    text="Central Medical Hospital",
    bbox=bbox1,
    label="HEADER"
)
entities.append(entity1)

# Question entity
bbox2 = BBoxModel(x1=50, y1=120, x2=150, y2=140)
entity2 = XFUNDEntity(
    id=1,
    text="Patient Name:",
    bbox=bbox2,
    label="QUESTION"
)
entities.append(entity2)

# Answer entity
bbox3 = BBoxModel(x1=160, y1=120, x2=300, y2=140)
entity3 = XFUNDEntity(
    id=2,
    text="John Smith",
    bbox=bbox3,
    label="ANSWER"
)
entities.append(entity3)

# Create annotation
annotation = XFUNDAnnotation(
    form=entities,
    image_path="medical_form_001.png",
    image_id="med_001"
)

print(f"Created annotation with {len(annotation.form)} entities")
```

## Example 9: Configuration Variations

### Medical Documents

```python
medical_config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/medical.csv",
    output_dir="output/medical",
    document_type=DocumentType.MEDICAL,
    enable_augmentations=True,
    augmentation_difficulty=AugmentationDifficulty.MEDIUM
)
```

### Clean Forms

```python
form_config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/forms.csv",
    output_dir="output/forms",
    document_type=DocumentType.FORM,
    enable_augmentations=False,
    image_dpi=600,
    add_bbox_jitter=False
)
```

### Research Dataset

```python
research_config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/research.csv",
    output_dir="output/research",
    enable_augmentations=True,
    augmentation_difficulty=AugmentationDifficulty.HARD,
    strict_validation=True,
    generate_debug_overlays=True
)
```

## Example 10: Error Handling

### Graceful Error Handling

```python
from src.models import GeneratorConfig
from src.generate_dataset import XFUNDGenerator
from pydantic import ValidationError

try:
    # Create configuration
    config = GeneratorConfig(
        templates_dir="data/templates_docx",
        csv_path="data/csv/medical.csv",
        output_dir="output/test"
    )
    
    # Generate dataset
    generator = XFUNDGenerator(config)
    result = generator.generate_dataset()
    
    if result.success:
        print(f"✅ Success! Generated {result.generated_entries} entries")
    else:
        print("❌ Generation failed:")
        for error in result.errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
            
except ValidationError as e:
    print(f"❌ Configuration error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
```

## Running the Examples

### Save as Script

Create a file `basic_example.py`:

```python
#!/usr/bin/env python3
"""Basic XFUND Generator example."""

from src.models import GeneratorConfig, DocumentType
from src.generate_dataset import XFUNDGenerator

def main():
    # Simple configuration
    config = GeneratorConfig(
        templates_dir="data/templates_docx",
        csv_path="data/csv/data.csv",
        output_dir="output/basic_example",
        document_type=DocumentType.MEDICAL,
        enable_augmentations=True
    )
    
    # Generate dataset
    generator = XFUNDGenerator(config)
    result = generator.generate_dataset()
    
    # Show results
    print(f"Generated: {result.generated_entries}")
    print(f"Failed: {result.failed_entries}")
    print(f"Success rate: {result.success_rate:.1f}%")
    
    if result.errors:
        print("Errors encountered:")
        for error in result.errors[:3]:
            print(f"  - {error}")

if __name__ == "__main__":
    main()
```

### Run the Script

```bash
python basic_example.py
```

## Tips for Beginners

1. **Start Simple**: Begin with small datasets and no augmentations
2. **Validate First**: Use `--validate-only` to check configuration
3. **Check Output**: Always review generated images and annotations
4. **Handle Errors**: Use try-catch blocks for robust code
5. **Use Type Hints**: Leverage Pydantic for better error messages

## Next Steps

- Learn [Advanced Examples](advanced.md) for complex scenarios
- Read [Configuration Guide](../configuration.md) for all options
- Explore [Pydantic Integration](../pydantic_integration.md) for type safety

---

*These examples provide a solid foundation. Experiment with different configurations to find what works best for your use case!*