# API Reference

This document provides comprehensive API documentation for the XFUND Generator.

## Core Classes

### GeneratorConfig

Configuration model for the XFUND dataset generator.

**Location:** `xfund_generator.models.GeneratorConfig`

```python
from xfund_generator.models import GeneratorConfig

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
| `fonts_dir` | `Optional[str]` | No | `None` | Directory containing custom fonts |
| `image_dpi` | `int` | No | `300` | DPI for generated images (72-600) |
| `target_size` | `int` | No | `1000` | Target size for XFUND normalization (min 224) |
| `document_type` | `DocumentType` | No | `MEDICAL` | Type of documents to generate |
| `enable_augmentations` | `bool` | No | `True` | Enable data augmentations |
| `augmentation_difficulty` | `AugmentationDifficulty` | No | `MEDIUM` | Augmentation difficulty preset |
| `add_bbox_jitter` | `bool` | No | `True` | Add small random jitter to bounding boxes |
| `strict_validation` | `bool` | No | `False` | Enable strict validation |
| `strict_augmentation` | `bool` | No | `False` | Enable strict augmentation validation |
| `generate_debug_overlays` | `bool` | No | `False` | Generate debug overlay images |
| `max_linking_distance` | `int` | No | `100` | Maximum pixel distance for Q/A linking |
| `max_linked_answers` | `int` | No | `3` | Maximum answers to link per question |
| `max_workers` | `int` | No | `4` | Maximum worker threads |
| `batch_size` | `int` | No | `10` | Batch size for processing |

#### Validators

- **templates_dir**: Path is resolved to absolute
- **csv_path**: Path is resolved to absolute
- **output_dir**: Will be created if doesn't exist
- **image_dpi**: Must be between 72 and 600
- **max_workers**: Must be >= 1
- **batch_size**: Must be >= 1

### DataRecord

Model representing a single data record from CSV.

**Location:** `xfund_generator.models.DataRecord`

```python
from xfund_generator.models import DataRecord

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
| `bbox` | `str` | No | Bounding box as "x1,y1,x2,y2" |

#### Methods

##### `get_bbox_coordinates() -> tuple[int, int, int, int]`

Parse bbox string into coordinates. Returns `(0, 0, 0, 0)` if bbox is empty or invalid.

```python
x1, y1, x2, y2 = record.get_bbox_coordinates()
```

##### `get_field(field_name: str, default: str = "") -> str`

Get a field value by name, checking both standard fields and additional_fields.

### XFUNDGenerator

Main generator class for creating XFUND datasets.

**Location:** `xfund_generator.generate_dataset.XFUNDGenerator`

```python
from xfund_generator.generate_dataset import XFUNDGenerator
from xfund_generator.models import GeneratorConfig

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

### BBoxModel

Bounding box model with validation and coordinate transformations.

**Location:** `xfund_generator.models.BBoxModel`

```python
from xfund_generator.models import BBoxModel

bbox = BBoxModel(x1=100.0, y1=50.0, x2=300.0, y2=80.0)
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `x1` | `float` | Yes | Left coordinate (must be >= 0) |
| `y1` | `float` | Yes | Top coordinate (must be >= 0) |
| `x2` | `float` | Yes | Right coordinate (must be > x1) |
| `y2` | `float` | Yes | Bottom coordinate (must be > y1) |

#### Methods

##### `to_list() -> list[float]`

Convert to coordinate list `[x1, y1, x2, y2]` as floats.

##### `to_xfund_format() -> list[int]`

Convert to XFUND integer format `[x1, y1, x2, y2]` with rounding.

##### `normalize(img_width, img_height, target_size=1000) -> BBoxModel`

Scale from image coordinates to normalized (0-target_size) scale.

```python
normalized = bbox.normalize(img_width=2480, img_height=3508, target_size=1000)
```

##### `denormalize(img_width, img_height, source_size=1000) -> BBoxModel`

Convert from normalized (0-source_size) scale back to image coordinates.

```python
actual = normalized.denormalize(img_width=2480, img_height=3508)
```

##### `area() -> float`

Calculate bounding box area.

##### `center() -> tuple[float, float]`

Calculate bounding box center point `(cx, cy)`.

### AugmentationConfig

Configuration for document augmentations.

**Location:** `xfund_generator.models.AugmentationConfig`

```python
from xfund_generator.models import AugmentationConfig

config = AugmentationConfig(
    enable_noise=True,
    enable_blur=True,
    augmentation_probability=0.7,
    min_visibility=0.3,
)
```

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `difficulty` | `AugmentationDifficulty` | `MEDIUM` | Difficulty preset |
| `document_type` | `DocumentType` | `MEDICAL` | Target document type |
| `enable_noise` | `bool` | `True` | Enable noise augmentations |
| `enable_blur` | `bool` | `True` | Enable blur augmentations |
| `enable_brightness` | `bool` | `True` | Enable brightness/contrast |
| `enable_rotation` | `bool` | `True` | Enable rotation |
| `enable_perspective` | `bool` | `True` | Enable perspective transform |
| `augmentation_probability` | `float` | `0.7` | Probability of applying augmentations (0.0-1.0) |
| `scanning_artifacts` | `bool` | `False` | Add scanning artifact effects |
| `paper_effects` | `bool` | `False` | Add paper texture and fold effects |
| `ink_bleeding` | `bool` | `False` | Simulate ink bleeding |
| `target_size` | `int` | `1000` | Target size for bbox normalization |
| `min_visibility` | `float` | `0.3` | Min visible area ratio to keep bbox after augmentation |

#### Class Methods

##### `from_yaml(file_path) -> AugmentationConfig`

Load augmentation config from a YAML file.

##### `to_yaml(file_path) -> None`

Save augmentation config to a YAML file.

##### `from_difficulty(difficulty, document_type) -> AugmentationConfig`

Create config from difficulty preset ("easy", "medium", "hard", "extreme").

### XFUNDEntity

Entity model for XFUND annotations.

**Location:** `xfund_generator.models.XFUNDEntity`

```python
from xfund_generator.models import XFUNDEntity, BBoxModel

entity = XFUNDEntity(
    id=1,
    text="John Doe",
    bbox=BBoxModel(x1=100, y1=50, x2=300, y2=80),
    label="ANSWER"
)
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `int` | Yes | Entity ID |
| `text` | `str` | Yes | Entity text content (non-empty) |
| `bbox` | `BBoxModel` | Yes | Bounding box |
| `label` | `str` | Yes | Entity label: HEADER, QUESTION, ANSWER, or OTHER |
| `words` | `list[dict]` | No | Word-level annotations |
| `linking` | `list[list[int]]` | No | Q/A linking pairs |

### TemplateValidationResult

Result model for DOCX template validation.

**Location:** `xfund_generator.models.TemplateValidationResult`

```python
from xfund_generator.docx_utils import validate_docx_template

result = validate_docx_template("medical_form.docx")
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
| `placeholders` | `list[str]` | Yes | Found placeholders in template |
| `paragraph_count` | `int` | Yes | Number of paragraphs in template |
| `table_count` | `int` | Yes | Number of tables in template |

#### Class Methods

##### `create_error(error_message: str) -> TemplateValidationResult`

Create a failed validation result.

##### `create_success(placeholders, paragraph_count, table_count) -> TemplateValidationResult`

Create a successful validation result.

### WordAnnotation

Word-level annotation model with validation.

**Location:** `xfund_generator.models.WordAnnotation`

```python
from xfund_generator.models import WordAnnotation

word = WordAnnotation(
    text="John",
    bbox=[100, 50, 200, 80],
    label="ANSWER"
)
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `str` | Yes | Word text (non-empty) |
| `bbox` | `list[int]` | Yes | Bounding box [x1, y1, x2, y2] (4 integers, x1 < x2, y1 < y2) |
| `label` | `str` | Yes | Label for this word |

## Form Classes

The `xfund_generator.form` package provides format-specific annotation classes.

### BaseAnnotation / BaseDataset

Base classes for all annotation formats.

```python
from xfund_generator.form import Word, XFUNDAnnotation, XFUNDDataset
```

### XFUNDDataset

XFUND format with question-answer linking.

```python
dataset = XFUNDDataset(annotations=[...])
json_output = dataset.to_json()  # Unified JSON export

# Get Q/A pairs
pairs = dataset.get_flat_qa_pairs()
grouped = dataset.get_grouped_qa_pairs()
```

### FUNSDDataset / WildReceiptDataset

Alternative annotation formats with their own export specializations.

## Utility Functions

### load_csv_data_as_models

Load CSV data as validated Pydantic models.

**Location:** `xfund_generator.utils.load_csv_data_as_models`

```python
from xfund_generator.utils import load_csv_data_as_models

records = load_csv_data_as_models("data/csv/data.csv")
```

### save_xfund_annotation_pydantic

Save XFUND annotation using Pydantic model.

**Location:** `xfund_generator.utils.save_xfund_annotation_pydantic`

```python
from xfund_generator.utils import save_xfund_annotation_pydantic

save_xfund_annotation_pydantic(annotation, "output/annotations/0001.json")
```

### validate_docx_template

Validate DOCX template and extract information.

**Location:** `xfund_generator.docx_utils.validate_docx_template`

```python
from xfund_generator.docx_utils import validate_docx_template

result = validate_docx_template("template.docx")
if result.valid:
    print(f"Found {len(result.placeholders)} placeholders")
```

### validate_config_file

Validate configuration file.

**Location:** `xfund_generator.models.validate_config_file`

```python
from xfund_generator.models import validate_config_file

result = validate_config_file("config.json")
if result.is_valid:
    print("Configuration is valid")
else:
    for error in result.errors:
        print(f"Error: {error}")
```

### normalize_field_name

Normalize field names to standard format. Supports custom field mappings.

**Location:** `xfund_generator.utils.normalize_field_name`

```python
from xfund_generator.utils import normalize_field_name

# Uses default medical field mappings
name = normalize_field_name("Doctor Name")  # -> "doctor_name"

# Use custom mappings
custom = {"invoice_number": ["invoice no", "inv_num", "invoice #"]}
name = normalize_field_name("Invoice No", custom_mappings=custom)
```

## Result Models

### GenerationResult

Results from dataset generation.

| Field | Type | Description |
|-------|------|-------------|
| `generated_entries` | `int` | Number of entries successfully generated |
| `failed_entries` | `int` | Number of entries that failed |
| `errors` | `list[str]` | List of error messages |
| `output_paths` | `list[str]` | List of output file paths |
| `processing_time` | `float` | Total processing time in seconds |

### ValidationResult

Results from validation operations.

| Field | Type | Description |
|-------|------|-------------|
| `is_valid` | `bool` | Whether validation passed |
| `errors` | `list[str]` | List of validation errors |
| `warnings` | `list[str]` | List of validation warnings |

## Constants

### Entity Labels

Standard XFUND entity labels:

```python
VALID_LABELS = ["HEADER", "QUESTION", "ANSWER", "OTHER"]
```

## Usage Patterns

### Basic Usage

```python
from xfund_generator.models import GeneratorConfig
from xfund_generator.generate_dataset import XFUNDGenerator

# Load configuration
config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/data.csv",
    output_dir="output",
)

# Create generator and generate dataset
generator = XFUNDGenerator(config)
result = generator.generate_dataset()
print(f"Generated {result.generated_entries} entries")
```

### Error Handling

```python
from pydantic import ValidationError

try:
    config = GeneratorConfig(
        templates_dir="data/templates_docx",
        csv_path="data/csv/data.csv",
        output_dir="output",
    )
    generator = XFUNDGenerator(config)
    result = generator.generate_dataset()
except ValidationError as e:
    print(f"Configuration validation failed: {e}")
except FileNotFoundError as e:
    print(f"Required file not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

For complete type information, refer to the source code in `xfund_generator/models.py` and related modules.
