# Configuration

Complete reference for XFUND Generator configuration options.

## Configuration Methods

### JSON File

```json
{
  "templates_dir": "data/templates_docx",
  "csv_path": "data/csv/data.csv",
  "output_dir": "output"
}
```

```bash
xfund-generator --config config.json
```

### Python API

```python
from xfund_generator import GeneratorConfig

config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/data.csv",
    output_dir="output"
)
```

## Configuration Options

### Required Options

| Option | Type | Description |
|--------|------|-------------|
| `templates_dir` | string | Path to DOCX template directory |
| `csv_path` | string | Path to CSV data file |
| `output_dir` | string | Output directory for generated files |

### Document Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `document_type` | string | `"medical"` | Document type (see below) |
| `fonts_dir` | string | `"fonts/handwritten_fonts"` | Custom fonts directory |

**Document Types:**
- `medical` - Medical reports and forms
- `form` - General forms and applications
- `invoice` - Invoices and receipts
- `contract` - Contracts and legal documents
- `general` - General documents

### Image Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `image_dpi` | integer | `300` | Image resolution in DPI |
| `target_size` | integer | `1000` | Target image size in pixels |
| `add_bbox_jitter` | boolean | `true` | Add random jitter to bounding boxes |

### Augmentation Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_augmentations` | boolean | `true` | Enable image augmentations |
| `augmentation_difficulty` | string | `"medium"` | Difficulty level (see below) |
| `strict_augmentation` | boolean | `false` | Fail on augmentation errors |

**Augmentation Levels:**

| Level | Effects |
|-------|---------|
| `easy` | Light brightness/contrast adjustments |
| `medium` | Blur, noise, slight rotation |
| `hard` | Perspective distortion, heavy noise |
| `extreme` | Aggressive transformations |

### Validation Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `strict_validation` | boolean | `true` | Enable strict validation mode |
| `generate_debug_overlays` | boolean | `true` | Generate debug overlay images |

### Performance Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_workers` | integer | `4` | Maximum worker threads |
| `batch_size` | integer | `10` | Batch size for processing |

## Example Configurations

### Minimal Configuration

```json
{
  "templates_dir": "data/templates_docx",
  "csv_path": "data/csv/data.csv",
  "output_dir": "output"
}
```

### Production Configuration

```json
{
  "templates_dir": "data/templates_docx",
  "csv_path": "data/csv/data.csv",
  "output_dir": "output",
  "document_type": "medical",
  "enable_augmentations": true,
  "augmentation_difficulty": "medium",
  "image_dpi": 300,
  "target_size": 1000,
  "strict_validation": true,
  "generate_debug_overlays": false,
  "max_workers": 8,
  "batch_size": 20
}
```

### High-Quality Configuration

```json
{
  "templates_dir": "data/templates_docx",
  "csv_path": "data/csv/data.csv",
  "output_dir": "output",
  "image_dpi": 600,
  "target_size": 2000,
  "enable_augmentations": false,
  "strict_validation": true,
  "generate_debug_overlays": true
}
```

### Training Data Configuration

```json
{
  "templates_dir": "data/templates_docx",
  "csv_path": "data/csv/data.csv",
  "output_dir": "output",
  "enable_augmentations": true,
  "augmentation_difficulty": "hard",
  "add_bbox_jitter": true,
  "max_workers": 16,
  "batch_size": 50
}
```

## Environment Variables

Override configuration with environment variables:

```bash
export XFUND_DATA_DIR=/path/to/data
export XFUND_OUTPUT_DIR=/path/to/output
```

## Validation

Validate configuration without generating:

```bash
xfund-generator --config config.json --validate-only
```

```python
config = GeneratorConfig(...)
validation_result = config.validate()
if validation_result.is_valid:
    print("Configuration valid")
else:
    print(f"Errors: {validation_result.errors}")
```

## See Also

- [[Getting-Started]] - Basic usage
- [[Annotation-Formats]] - Output formats
- [[API-Reference]] - Full API docs
