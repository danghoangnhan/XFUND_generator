# API Reference

Complete API documentation for XFUND Generator.

## Core Classes

### GeneratorConfig

Configuration model for the generator.

```python
from xfund_generator import GeneratorConfig

config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/data.csv",
    output_dir="output",
    document_type="medical",
    enable_augmentations=True,
    augmentation_difficulty="medium",
    image_dpi=300,
    target_size=1000,
    max_workers=4,
    batch_size=10
)
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `templates_dir` | str | Yes | - | Path to DOCX templates |
| `csv_path` | str | Yes | - | Path to CSV data |
| `output_dir` | str | Yes | - | Output directory |
| `fonts_dir` | str | No | `"fonts/handwritten_fonts"` | Fonts directory |
| `document_type` | str | No | `"medical"` | Document type |
| `enable_augmentations` | bool | No | `True` | Enable augmentations |
| `augmentation_difficulty` | str | No | `"medium"` | Augmentation level |
| `image_dpi` | int | No | `300` | Image DPI |
| `target_size` | int | No | `1000` | Target image size |
| `add_bbox_jitter` | bool | No | `True` | Add bbox jitter |
| `strict_validation` | bool | No | `True` | Strict validation |
| `strict_augmentation` | bool | No | `False` | Strict augmentation |
| `generate_debug_overlays` | bool | No | `True` | Generate debug images |
| `max_workers` | int | No | `4` | Worker threads |
| `batch_size` | int | No | `10` | Batch size |

### XFUNDGenerator

Main generator class.

```python
from xfund_generator import XFUNDGenerator

generator = XFUNDGenerator(config)
result = generator.generate_dataset()
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate_dataset()` | None | `GenerationResult` | Generate full dataset |
| `generate_single(record)` | `DataRecord` | `SingleResult` | Generate single document |
| `validate_config()` | None | `ValidationResult` | Validate configuration |

### GenerationResult

Result of dataset generation.

```python
result = generator.generate_dataset()

print(result.generated_entries)  # Number of entries
print(result.output_path)        # Output directory
print(result.errors)             # List of errors
print(result.warnings)           # List of warnings
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `generated_entries` | int | Number of entries generated |
| `output_path` | str | Path to output directory |
| `errors` | list[str] | List of error messages |
| `warnings` | list[str] | List of warning messages |
| `duration_seconds` | float | Generation duration |

## Model Classes

### BBoxModel

Bounding box with validation.

```python
from xfund_generator import BBoxModel

bbox = BBoxModel(x1=10, y1=20, x2=100, y2=80)

print(bbox.width)   # 90
print(bbox.height)  # 60
print(bbox.area)    # 5400
print(bbox.center)  # (55.0, 50.0)
print(bbox.to_list())  # [10, 20, 100, 80]
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `x1` | float | Left coordinate |
| `y1` | float | Top coordinate |
| `x2` | float | Right coordinate |
| `y2` | float | Bottom coordinate |

**Computed Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `width` | float | Box width |
| `height` | float | Box height |
| `area` | float | Box area |
| `center` | tuple | Center point (x, y) |

### DataRecord

Data record for template population.

```python
from xfund_generator import DataRecord

record = DataRecord(
    patient_name="John Smith",
    doctor_name="Dr. Johnson",
    date="2024-01-15"
)
```

## Form Classes

### Base Classes

#### Word

Word-level annotation.

```python
from xfund_generator.form.base import Word

word = Word(text="Patient", box=[50, 100, 100, 120])
```

#### BaseAnnotation

Base class for all annotations.

```python
from xfund_generator.form.base import BaseAnnotation

# Abstract - use format-specific classes
```

#### BaseDataset

Base class for all datasets.

```python
from xfund_generator.form.base import BaseDataset

# Abstract - use format-specific classes
```

### XFUND Classes

```python
from xfund_generator.form.xfund import XFUNDDataset, XFUNDAnnotation

dataset = XFUNDDataset(image_path="doc.png")

annotation = XFUNDAnnotation(
    id=0,
    text="Patient Name:",
    box=[50, 100, 150, 120],
    label="question",
    words=[Word(text="Patient", box=[50, 100, 100, 120])],
    linking=[[0, 1]]
)

dataset.add_annotation(annotation)
output = dataset.to_json()
```

### FUNSD Classes

```python
from xfund_generator.form.funsd import FUNSDDataset, FUNSDAnnotation

dataset = FUNSDDataset(image_path="doc.png")

annotation = FUNSDAnnotation(
    id=0,
    text="Patient Name:",
    box=[50, 100, 150, 120],
    label="question",
    words=[Word(text="Patient", box=[50, 100, 100, 120])],
    key_id=0,
    value_id=1
)

dataset.add_annotation(annotation)
output = dataset.to_json()
```

### WildReceipt Classes

```python
from xfund_generator.form.wildreceipt import WildReceiptDataset, WildReceiptAnnotation

dataset = WildReceiptDataset(image_path="doc.png")

annotation = WildReceiptAnnotation(
    id=0,
    text="Total: $100",
    box=[50, 100, 150, 120],
    label="total",
    words=[Word(text="Total:", box=[50, 100, 90, 120])]
)

dataset.add_annotation(annotation)
output = dataset.to_json()
```

## Augmentation

### DocumentAugmenter

```python
from xfund_generator.augmentations import DocumentAugmenter

augmenter = DocumentAugmenter(
    brightness_range=(0.7, 1.3),
    blur_probability=0.3,
    noise_probability=0.2
)

augmented_image = augmenter.apply(image)
```

## Utilities

### validate_annotation_quality

```python
from xfund_generator.utils import validate_annotation_quality

issues = validate_annotation_quality(annotation)
if issues:
    print("Quality issues:", issues)
```

## CLI Reference

```bash
# Basic usage
xfund-generator --config config.json

# Validate only
xfund-generator --config config.json --validate-only

# Show help
xfund-generator --help

# Show version
xfund-generator --version
```

## See Also

- [[Configuration]] - Configuration options
- [[Annotation-Formats]] - Format specifications
- [[Testing]] - Testing guide
