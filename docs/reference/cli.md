# CLI Reference

Command-line interface reference for the XFUND Generator.

## Overview

The XFUND Generator provides several command-line scripts for different tasks:

- **Dataset Generation**: Create XFUND datasets from templates and data
- **OCR Evaluation**: Evaluate OCR model performance 
- **Configuration Management**: Create and validate configurations
- **Debugging Tools**: Debug templates and data issues

## Dataset Generation

### generate_dataset.py

Main script for generating XFUND datasets.

```bash
python src/generate_dataset.py [OPTIONS]
```

#### Options

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--config` | `str` | No | `config.json` | Path to configuration file |
| `--output-dir` | `str` | No | - | Override output directory |
| `--templates-dir` | `str` | No | - | Override templates directory |
| `--csv-path` | `str` | No | - | Override CSV data path |
| `--dry-run` | `flag` | No | `False` | Validate without generating |
| `--verbose` | `flag` | No | `False` | Enable verbose output |
| `--debug` | `flag` | No | `False` | Enable debug mode |
| `--workers` | `int` | No | `4` | Number of worker threads |

#### Examples

**Basic generation:**
```bash
python src/generate_dataset.py --config config.json
```

**Custom paths:**
```bash
python src/generate_dataset.py \
    --templates-dir "custom/templates" \
    --csv-path "custom/data.csv" \
    --output-dir "custom/output" \
    --verbose
```

**Dry run validation:**
```bash
python src/generate_dataset.py --config config.json --dry-run
```

**Debug mode:**
```bash
python src/generate_dataset.py --config config.json --debug --verbose
```

#### Exit Codes

- `0`: Success
- `1`: General error
- `2`: Configuration error
- `3`: Template error
- `4`: Data error

### Demo Scripts

#### demo_pydantic_integration.py

Demonstrate Pydantic integration features.

```bash
python demo_pydantic_integration.py [OPTIONS]
```

**Options:**
- `--config`: Configuration file path
- `--example`: Run specific example (config|data|generation|validation)

**Examples:**
```bash
# Run all examples
python demo_pydantic_integration.py

# Run specific example
python demo_pydantic_integration.py --example validation

# Use custom config
python demo_pydantic_integration.py --config custom_config.json
```

#### run_extraction_examples.py

Run structured extraction examples.

```bash
python run_extraction_examples.py [OPTIONS]
```

**Options:**
- `--input-dir`: Input directory with documents
- `--output-dir`: Output directory for results
- `--model`: Extraction model to use

## OCR Evaluation

### evaluate_ocr.py

Evaluate OCR model performance.

```bash
python evaluate_ocr.py [OPTIONS]
```

#### Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--ground-truth` | `str` | Yes | Ground truth annotations path |
| `--predictions` | `str` | Yes | OCR predictions path |
| `--output` | `str` | No | Output results path |
| `--format` | `str` | No | Output format (json|csv|xlsx) |
| `--metrics` | `str` | No | Metrics to calculate (all|precision|recall|f1) |

#### Examples

**Basic evaluation:**
```bash
python evaluate_ocr.py \
    --ground-truth data/ground_truth.json \
    --predictions data/predictions.json \
    --output results/evaluation.json
```

**Custom metrics:**
```bash
python evaluate_ocr.py \
    --ground-truth data/ground_truth.json \
    --predictions data/predictions.json \
    --metrics "precision,recall,f1" \
    --format xlsx
```

### compare_ocr_models.py

Compare multiple OCR models.

```bash
python compare_ocr_models.py [OPTIONS]
```

#### Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--models-dir` | `str` | Yes | Directory containing model results |
| `--ground-truth` | `str` | Yes | Ground truth annotations |
| `--output` | `str` | No | Comparison output file |
| `--plot` | `flag` | No | Generate comparison plots |

#### Examples

**Compare models:**
```bash
python compare_ocr_models.py \
    --models-dir evaluation_results/ \
    --ground-truth data/ground_truth.json \
    --output model_comparison.json \
    --plot
```

## Configuration Management

### create_config.py

Create new configuration files.

```bash
python scripts/create_config.py [OPTIONS]
```

#### Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--template` | `str` | No | Configuration template (basic|advanced|medical) |
| `--output` | `str` | No | Output configuration file |
| `--interactive` | `flag` | No | Interactive configuration setup |

#### Examples

**Create basic config:**
```bash
python scripts/create_config.py --template basic --output basic_config.json
```

**Interactive setup:**
```bash
python scripts/create_config.py --interactive
```

### validate_config.py

Validate configuration files.

```bash
python scripts/validate_config.py CONFIG_FILE
```

#### Examples

**Validate configuration:**
```bash
python scripts/validate_config.py config.json
```

**Validate with verbose output:**
```bash
python scripts/validate_config.py config.json --verbose
```

## Debugging Tools

### debug_templates.py

Debug template issues.

```bash
python scripts/debug_templates.py [OPTIONS]
```

#### Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--templates-dir` | `str` | Yes | Templates directory |
| `--template` | `str` | No | Specific template to debug |
| `--check-layouts` | `flag` | No | Check layout JSON files |
| `--test-conversion` | `flag` | No | Test DOCX to image conversion |

#### Examples

**Debug all templates:**
```bash
python scripts/debug_templates.py --templates-dir data/templates_docx
```

**Debug specific template:**
```bash
python scripts/debug_templates.py \
    --templates-dir data/templates_docx \
    --template medical_form \
    --check-layouts \
    --test-conversion
```

### debug_data.py

Debug CSV data issues.

```bash
python scripts/debug_data.py [OPTIONS]
```

#### Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--csv-path` | `str` | Yes | CSV data file path |
| `--validate-records` | `flag` | No | Validate individual records |
| `--check-encoding` | `flag` | No | Check file encoding |
| `--sample-size` | `int` | No | Number of records to check |

#### Examples

**Debug CSV data:**
```bash
python scripts/debug_data.py \
    --csv-path data/csv/data.csv \
    --validate-records \
    --check-encoding
```

**Check sample:**
```bash
python scripts/debug_data.py \
    --csv-path data/csv/data.csv \
    --sample-size 10
```

## Environment Variables

The CLI tools support several environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `XFUND_CONFIG_PATH` | Default configuration file | `config.json` |
| `XFUND_OUTPUT_DIR` | Default output directory | `output` |
| `XFUND_TEMPLATES_DIR` | Default templates directory | `data/templates_docx` |
| `XFUND_DATA_PATH` | Default CSV data path | `data/csv/data.csv` |
| `XFUND_LOG_LEVEL` | Logging level | `INFO` |
| `XFUND_MAX_WORKERS` | Maximum worker threads | `4` |

### Setting Environment Variables

**Linux/macOS:**
```bash
export XFUND_CONFIG_PATH="custom_config.json"
export XFUND_LOG_LEVEL="DEBUG"
```

**Windows:**
```cmd
set XFUND_CONFIG_PATH=custom_config.json
set XFUND_LOG_LEVEL=DEBUG
```

**Using .env file:**
```bash
# Create .env file
cat > .env << EOF
XFUND_CONFIG_PATH=config.json
XFUND_OUTPUT_DIR=output
XFUND_LOG_LEVEL=INFO
EOF

# Load with python-dotenv
pip install python-dotenv
```

## Logging Configuration

### Log Levels

- `DEBUG`: Detailed debug information
- `INFO`: General information messages
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical errors

### Log Formats

**Console output:**
```
2024-11-09 10:30:15 INFO: Starting dataset generation...
2024-11-09 10:30:16 DEBUG: Loading configuration from config.json
2024-11-09 10:30:17 INFO: Found 5 templates
```

**File output:**
```bash
# Enable file logging
python src/generate_dataset.py --config config.json --log-file generation.log
```

### Custom Logging

```python
import logging

# Configure custom logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xfund.log'),
        logging.StreamHandler()
    ]
)
```

## Configuration File Examples

### Basic CLI Config

```json
{
  "templates_dir": "data/templates_docx",
  "csv_path": "data/csv/data.csv", 
  "output_dir": "output",
  "image_dpi": 300,
  "max_workers": 4,
  "enable_augmentations": false,
  "strict_validation": true
}
```

### Advanced CLI Config

```json
{
  "templates_dir": "data/templates_docx",
  "csv_path": "data/csv/data.csv",
  "output_dir": "output",
  "image_dpi": 600,
  "max_workers": 8,
  "batch_size": 20,
  "enable_augmentations": true,
  "generate_debug_overlays": true,
  "strict_validation": true,
  "augmentation_config": {
    "rotation_range": 5,
    "noise_level": 0.1,
    "brightness_range": 0.2
  }
}
```

## Batch Processing

### Process Multiple Configs

```bash
#!/bin/bash
# Process multiple configurations

configs=("config1.json" "config2.json" "config3.json")

for config in "${configs[@]}"; do
    echo "Processing $config..."
    python src/generate_dataset.py --config "$config"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $config"
    else
        echo "✗ Failed to process $config"
    fi
done
```

### Parallel Processing

```bash
#!/bin/bash
# Process configurations in parallel

python src/generate_dataset.py --config config1.json &
python src/generate_dataset.py --config config2.json &
python src/generate_dataset.py --config config3.json &

# Wait for all to complete
wait
echo "All processing complete"
```

## Performance Monitoring

### Resource Usage

```bash
# Monitor resource usage during generation
htop &
python src/generate_dataset.py --config config.json

# Or use built-in monitoring
python src/generate_dataset.py --config config.json --monitor-resources
```

### Progress Tracking

```bash
# Enable progress tracking
python src/generate_dataset.py --config config.json --progress

# Output:
# Processing templates... ████████████████████ 100%
# Generating images...   ████████████████████ 100%
# Creating annotations... ███████████████████ 100%
```

## Integration Examples

### Makefile Integration

```makefile
# Add to Makefile
.PHONY: generate-dataset validate-config debug-templates

generate-dataset:
	python src/generate_dataset.py --config config.json --verbose

validate-config:
	python scripts/validate_config.py config.json

debug-templates:
	python scripts/debug_templates.py --templates-dir data/templates_docx

evaluate-ocr:
	python evaluate_ocr.py \
		--ground-truth data/ground_truth.json \
		--predictions data/predictions.json \
		--output results/evaluation.json
```

### CI/CD Integration

```yaml
# GitHub Actions example
name: XFUND Dataset Generation
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        
    - name: Validate configuration
      run: |
        python scripts/validate_config.py config.json
        
    - name: Test generation (dry run)
      run: |
        python src/generate_dataset.py --config config.json --dry-run
```

## Error Codes Reference

| Code | Meaning | Description |
|------|---------|-------------|
| 0 | Success | Operation completed successfully |
| 1 | General Error | Unspecified error occurred |
| 2 | Configuration Error | Invalid or missing configuration |
| 3 | Template Error | Template loading or processing error |
| 4 | Data Error | CSV data loading or validation error |
| 5 | Generation Error | Error during dataset generation |
| 6 | OCR Error | Error in OCR evaluation |
| 10 | File Not Found | Required file not found |
| 11 | Permission Error | Insufficient permissions |
| 12 | Disk Space Error | Insufficient disk space |

---

*For more detailed information about specific functions and classes, see the [API Reference](api.md).*