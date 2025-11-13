# Configuration

This directory contains configuration files for the XFUND Generator.

## Example Configuration

### `example_config.json`

A complete example configuration with all available options:

```bash
# Use the example config
python src/generate_dataset.py --config config/example_config.json

# Validate config without generating
python src/generate_dataset.py --config config/example_config.json --validate-only
```

## Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `templates_dir` | string | `"data/templates_docx"` | Path to DOCX templates |
| `csv_path` | string | `"data/csv/data.csv"` | Path to CSV data file |
| `output_dir` | string | `"output"` | Output directory for generated files |
| `fonts_dir` | string | `"fonts/handwritten_fonts"` | Path to custom fonts |
| `document_type` | string | `"medical"` | Document type (`medical`, `form`, `invoice`, `contract`, `general`) |
| `enable_augmentations` | boolean | `true` | Enable image augmentations |
| `augmentation_difficulty` | string | `"medium"` | Augmentation level (`easy`, `medium`, `hard`, `extreme`) |
| `image_dpi` | integer | `300` | Image resolution in DPI |
| `target_size` | integer | `1000` | Target image size in pixels |
| `add_bbox_jitter` | boolean | `true` | Add random jitter to bounding boxes |
| `strict_validation` | boolean | `true` | Enable strict validation mode |
| `strict_augmentation` | boolean | `false` | Enable strict augmentation validation |
| `generate_debug_overlays` | boolean | `true` | Generate debug overlay images |
| `max_workers` | integer | `4` | Maximum number of worker threads |
| `batch_size` | integer | `10` | Batch size for processing |

## Creating Custom Configurations

Copy `example_config.json` and modify the values as needed:

```bash
cp config/example_config.json config/my_custom_config.json
# Edit my_custom_config.json
python src/generate_dataset.py --config config/my_custom_config.json
```

## Validation

All configurations are validated using Pydantic models to ensure type safety and correct values.