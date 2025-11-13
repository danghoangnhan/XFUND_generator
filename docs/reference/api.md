# API Reference

This document provides comprehensive API documentation for the XFUND Generator.

## Core Classes

### GeneratorConfig

Configuration model for the XFUND dataset generator.

**Location:** `src.models.GeneratorConfig`

```python
from src.models import GeneratorConfig

config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/data.csv",
    output_dir="output"
)
```

#### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `templates_dir` | `str` | Yes | - | Path to DOCX templates directory |
| `csv_path` | `str` | Yes | - | Path to CSV data file |
| `output_dir` | `str` | Yes | - | Output directory for generated files |
| `image_dpi` | `int` | No | `300` | DPI for generated images |
| `enable_augmentations` | `bool` | No | `False` | Enable data augmentations |
| `max_workers` | `int` | No | `4` | Maximum worker threads |
| `batch_size` | `int` | No | `10` | Batch size for processing |
| `strict_validation` | `bool` | No | `True` | Enable strict validation |
| `generate_debug_overlays` | `bool` | No | `False` | Generate debug overlays |

#### Class Methods

##### `from_json_file(file_path: str) -> GeneratorConfig`

Load configuration from JSON file.

```python
config = GeneratorConfig.from_json_file("config.json")
```

**Parameters:**
- `file_path` (str): Path to JSON configuration file

**Returns:**
- `GeneratorConfig`: Loaded configuration

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValidationError`: If configuration is invalid

##### `to_json_file(file_path: str) -> None`

Save configuration to JSON file.

```python
config.to_json_file("output_config.json")
```

**Parameters:**
- `file_path` (str): Output file path

#### Validators

- **templates_dir**: Must exist and be readable
- **csv_path**: Must exist and be readable
- **output_dir**: Will be created if doesn't exist
- **image_dpi**: Must be between 72 and 1200
- **max_workers**: Must be between 1 and 32
- **batch_size**: Must be greater than 0

### DataRecord

Model representing a single data record from CSV.

**Location:** `src.models.DataRecord`

```python
from src.models import DataRecord

record = DataRecord(
    template_name="medical_form",
    field_name="patient_name",
    field_value="John Doe",
    bbox="100,50,300,80"
)
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `template_name` | `str` | Yes | Name of template to use |
| `field_name` | `str` | Yes | Field identifier |
| `field_value` | `str` | Yes | Value to insert |
| `bbox` | `str` | Yes | Bounding box as "x1,y1,x2,y2" |

#### Methods

##### `get_bbox_coordinates() -> Tuple[int, int, int, int]`

Parse bbox string into coordinates.

```python
x1, y1, x2, y2 = record.get_bbox_coordinates()
```

**Returns:**
- `Tuple[int, int, int, int]`: (x1, y1, x2, y2) coordinates

**Raises:**
- `ValueError`: If bbox format is invalid

### XFUNDGenerator

Main generator class for creating XFUND datasets.

**Location:** `src.generate_dataset.XFUNDGenerator`

```python
from src.generate_dataset import XFUNDGenerator
from src.models import GeneratorConfig

config = GeneratorConfig(...)
generator = XFUNDGenerator(config)
```

#### Constructor

```python
def __init__(self, config: GeneratorConfig)
```

**Parameters:**
- `config` (GeneratorConfig): Configuration object

#### Methods

##### `generate_dataset() -> GenerationResult`

Generate complete XFUND dataset.

```python
result = generator.generate_dataset()
print(f"Generated: {result.generated_entries}")
print(f"Failed: {result.failed_entries}")
```

**Returns:**
- `GenerationResult`: Results of generation process

##### `validate_setup() -> ValidationResult`

Validate generator setup before generation.

```python
validation = generator.validate_setup()
if not validation.is_valid:
    for error in validation.errors:
        print(f"Error: {error}")
```

**Returns:**
- `ValidationResult`: Setup validation results

##### `get_templates() -> List[str]`

Get list of available templates.

```python
templates = generator.get_templates()
print(f"Available templates: {templates}")
```

**Returns:**
- `List[str]`: List of template names

### BBoxModel

Bounding box model with validation.

**Location:** `src.models.BBoxModel`

```python
from src.models import BBoxModel

bbox = BBoxModel(x1=100, y1=50, x2=300, y2=80)
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `x1` | `int` | Yes | Left coordinate |
| `y1` | `int` | Yes | Top coordinate |
| `x2` | `int` | Yes | Right coordinate |
| `y2` | `int` | Yes | Bottom coordinate |

#### Methods

##### `width() -> int`

Calculate bounding box width.

```python
w = bbox.width()
```

##### `height() -> int`

Calculate bounding box height.

```python
h = bbox.height()
```

##### `area() -> int`

Calculate bounding box area.

```python
area = bbox.area()
```

##### `to_list() -> List[int]`

Convert to coordinate list.

```python
coords = bbox.to_list()  # [x1, y1, x2, y2]
```

### XFUNDEntity

Entity model for XFUND annotations.

**Location:** `src.models.XFUNDEntity`

```python
from src.models import XFUNDEntity

entity = XFUNDEntity(
    text="John Doe",
    bbox=[100, 50, 300, 80],
    label="B-PERSON"
)
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `str` | Yes | Entity text content |
| `bbox` | `List[int]` | Yes | Bounding box coordinates [x1,y1,x2,y2] |
| `label` | `str` | Yes | Entity label |
| `words` | `List[Dict]` | No | Word-level annotations |

### TemplateValidationResult

Result model for DOCX template validation.

**Location:** `src.models.TemplateValidationResult`

```python
from src.models import TemplateValidationResult
from src.docx_utils import validate_docx_template

# Validate a template
result = validate_docx_template("medical_form.docx")

# Check result
if result.valid:
    print(f"Template is valid with {len(result.placeholders)} placeholders")
else:
    print(f"Template validation failed: {result.error}")
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `valid` | `bool` | Yes | Whether template is valid |
| `error` | `Optional[str]` | No | Error message if validation failed |
| `placeholders` | `List[str]` | Yes | Found placeholders in template |
| `paragraph_count` | `int` | Yes | Number of paragraphs in template |
| `table_count` | `int` | Yes | Number of tables in template |

#### Class Methods

##### `create_error(error_message: str) -> TemplateValidationResult`

Create a failed validation result.

```python
result = TemplateValidationResult.create_error("File not found")
```

##### `create_success(placeholders: List[str], paragraph_count: int, table_count: int) -> TemplateValidationResult`

Create a successful validation result.

```python
result = TemplateValidationResult.create_success(
    placeholders=["name", "date"],
    paragraph_count=5,
    table_count=2
)
```

### XFUNDAnnotation

Complete XFUND annotation model.

**Location:** `src.models.XFUNDAnnotation`

```python
from src.models import XFUNDAnnotation

annotation = XFUNDAnnotation(
    id=1,
    image_path="output/images/0001.png",
    entities=[entity1, entity2]
)
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `int` | Yes | Unique annotation ID |
| `image_path` | `str` | Yes | Path to associated image |
| `entities` | `List[XFUNDEntity]` | Yes | List of entities |
| `relations` | `List[Dict]` | No | Entity relations |

#### Methods

##### `to_xfund_format() -> Dict`

Convert to standard XFUND format.

```python
xfund_data = annotation.to_xfund_format()
```

## Utility Functions

### load_csv_data_as_models

Load CSV data as validated Pydantic models.

**Location:** `src.utils.load_csv_data_as_models`

```python
from src.utils import load_csv_data_as_models

records = load_csv_data_as_models("data/csv/data.csv")
```

**Parameters:**
- `csv_path` (str): Path to CSV file

**Returns:**
- `List[DataRecord]`: List of validated data records

**Raises:**
- `FileNotFoundError`: If CSV file not found
- `ValidationError`: If data validation fails

### save_xfund_annotation_pydantic

Save XFUND annotation using Pydantic model.

**Location:** `src.utils.save_xfund_annotation_pydantic`

```python
from src.utils import save_xfund_annotation_pydantic
from src.models import XFUNDAnnotation

save_xfund_annotation_pydantic(annotation, "output/annotations/0001.json")
```

**Parameters:**
- `annotation` (XFUNDAnnotation): Annotation to save
- `output_path` (str): Output file path

### validate_docx_template

Validate DOCX template and extract information.

**Location:** `src.docx_utils.validate_docx_template`

```python
from src.docx_utils import validate_docx_template

result = validate_docx_template("template.docx")
if result.valid:
    print(f"Found {len(result.placeholders)} placeholders")
    print(f"Paragraphs: {result.paragraph_count}")
    print(f"Tables: {result.table_count}")
else:
    print(f"Template validation failed: {result.error}")
```

**Parameters:**
- `template_path` (str): Path to DOCX template file

**Returns:**
- `TemplateValidationResult`: Validation result with template information

### validate_config_file

Validate configuration file.

**Location:** `src.models.validate_config_file`

```python
from src.models import validate_config_file

result = validate_config_file("config.json")
if result.is_valid:
    print("Configuration is valid")
else:
    for error in result.errors:
        print(f"Error: {error}")
```

**Parameters:**
- `file_path` (str): Path to configuration file

**Returns:**
- `ValidationResult`: Validation result with errors if any

### get_default_config

Get default configuration object.

**Location:** `src.models.get_default_config`

```python
from src.models import get_default_config

config = get_default_config()
config.csv_path = "my_data.csv"  # Customize as needed
```

**Returns:**
- `GeneratorConfig`: Default configuration

## Result Models

### GenerationResult

Results from dataset generation.

```python
class GenerationResult:
    generated_entries: int
    failed_entries: int
    errors: List[str]
    output_paths: List[str]
    processing_time: float
```

### ValidationResult

Results from validation operations.

```python
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
```

## Error Types

### ValidationError

Raised when Pydantic validation fails.

```python
from pydantic import ValidationError

try:
    config = GeneratorConfig(invalid_data)
except ValidationError as e:
    print(f"Validation error: {e}")
```

### ConfigurationError

Raised when configuration is invalid.

```python
from src.models import ConfigurationError

try:
    generator = XFUNDGenerator(config)
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### GenerationError

Raised when generation fails.

```python
from src.generate_dataset import GenerationError

try:
    result = generator.generate_dataset()
except GenerationError as e:
    print(f"Generation error: {e}")
```

## Constants

### Entity Labels

Standard XFUND entity labels:

```python
ENTITY_LABELS = [
    "B-HEADER",
    "I-HEADER", 
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER", 
    "I-ANSWER",
    "B-OTHER",
    "I-OTHER"
]
```

### Default Values

```python
DEFAULT_IMAGE_DPI = 300
DEFAULT_MAX_WORKERS = 4
DEFAULT_BATCH_SIZE = 10
MIN_BBOX_SIZE = 10
MAX_IMAGE_SIZE = 5000
```

## Usage Patterns

### Basic Usage

```python
from src.models import GeneratorConfig
from src.generate_dataset import XFUNDGenerator

# Load configuration
config = GeneratorConfig.from_json_file("config.json")

# Create generator
generator = XFUNDGenerator(config)

# Validate setup
validation = generator.validate_setup()
if not validation.is_valid:
    raise ValueError(f"Setup invalid: {validation.errors}")

# Generate dataset
result = generator.generate_dataset()
print(f"Generated {result.generated_entries} entries")
```

### Custom Configuration

```python
from src.models import GeneratorConfig

config = GeneratorConfig(
    templates_dir="custom/templates",
    csv_path="custom/data.csv", 
    output_dir="custom/output",
    image_dpi=600,
    enable_augmentations=True,
    strict_validation=True
)

# Save custom configuration
config.to_json_file("custom_config.json")
```

### Error Handling

```python
try:
    config = GeneratorConfig.from_json_file("config.json")
    generator = XFUNDGenerator(config)
    result = generator.generate_dataset()
except ValidationError as e:
    print(f"Configuration validation failed: {e}")
except FileNotFoundError as e:
    print(f"Required file not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Type Hints

The API uses comprehensive type hints for better IDE support:

```python
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

def generate_from_config(
    config_path: Union[str, Path],
    output_dir: Optional[str] = None
) -> GenerationResult:
    """Generate dataset from configuration file."""
    pass
```

For complete type information, refer to the source code in `src/models.py` and related modules.