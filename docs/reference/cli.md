# CLI Reference

Command-line interface reference for the XFUND Generator.

## Overview

The XFUND Generator can be invoked via:

```bash
# As a module
python -m xfund_generator [OPTIONS]

# Or directly
python xfund_generator/generate_dataset.py [OPTIONS]
```

## Dataset Generation

### generate_dataset.py

Main script for generating XFUND datasets.

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | `str` | - | Path to JSON configuration file |
| `--templates-dir` | `-t` | `str` | - | Override templates directory |
| `--csv-path` | `-d` | `str` | - | Override CSV data path |
| `--output-dir` | `-o` | `str` | - | Override output directory |
| `--no-augmentations` | - | `flag` | `False` | Disable data augmentations |
| `--debug` | - | `flag` | `False` | Enable debug mode with overlays |
| `--validate-only` | - | `flag` | `False` | Validate setup without generating |
| `--verbose` | `-v` | `flag` | `False` | Enable verbose output |

#### Examples

**Basic generation with config file:**
```bash
python -m xfund_generator --config config.json
```

**Custom paths (override config):**
```bash
python -m xfund_generator \
    --templates-dir "custom/templates" \
    --csv-path "custom/data.csv" \
    --output-dir "custom/output" \
    --verbose
```

**Validate setup without generating:**
```bash
python -m xfund_generator --config config.json --validate-only
```

**Debug mode (generates overlay images):**
```bash
python -m xfund_generator --config config.json --debug --verbose
```

**Disable augmentations:**
```bash
python -m xfund_generator --config config.json --no-augmentations
```

#### Exit Codes

| Code | Meaning | Description |
|------|---------|-------------|
| `0` | Success | Operation completed successfully |
| `1` | General Error | Unspecified error occurred |
| `2` | Configuration Error | Invalid or missing configuration |
| `3` | Template Error | Template loading or processing error |
| `4` | Data Error | CSV data loading or validation error |

## Configuration File Format

### Basic Config

```json
{
  "templates_dir": "data/templates_docx",
  "csv_path": "data/csv/data.csv",
  "output_dir": "output",
  "image_dpi": 300,
  "enable_augmentations": false
}
```

### Full Config

```json
{
  "templates_dir": "data/templates_docx",
  "csv_path": "data/csv/data.csv",
  "output_dir": "output",
  "image_dpi": 300,
  "target_size": 1000,
  "document_type": "medical",
  "enable_augmentations": true,
  "augmentation_difficulty": "medium",
  "add_bbox_jitter": true,
  "strict_validation": false,
  "generate_debug_overlays": false,
  "max_linking_distance": 100,
  "max_linked_answers": 3,
  "max_workers": 4,
  "batch_size": 10
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `XFUND_CONFIG_PATH` | Default configuration file | `config.json` |
| `XFUND_OUTPUT_DIR` | Default output directory | `output` |
| `XFUND_TEMPLATES_DIR` | Default templates directory | `data/templates_docx` |
| `XFUND_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

## Logging

### Log Levels

Control via `--verbose` flag or `XFUND_LOG_LEVEL` environment variable:

- `DEBUG`: Detailed debug information (enabled with `--verbose`)
- `INFO`: General progress messages (default)
- `WARNING`: Warning messages
- `ERROR`: Error messages

### Custom Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xfund.log'),
        logging.StreamHandler()
    ]
)
```

## Batch Processing

### Process Multiple Configs

```bash
#!/bin/bash
for config in configs/*.json; do
    echo "Processing $config..."
    python -m xfund_generator --config "$config"
    if [ $? -eq 0 ]; then
        echo "Successfully processed $config"
    else
        echo "Failed to process $config"
    fi
done
```

---

*For API details, see the [API Reference](api.md).*
