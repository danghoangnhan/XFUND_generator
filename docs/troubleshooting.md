# Troubleshooting Guide

This guide helps you resolve common issues when using the XFUND Generator.

## Installation Issues

### Problem: ImportError: No module named 'pydantic'

**Symptoms:**
```
ImportError: No module named 'pydantic'
ModuleNotFoundError: No module named 'pydantic'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows

# Install pydantic
pip install pydantic>=2.0.0

# Or reinstall all requirements
pip install -r requirements.txt
```

### Problem: LibreOffice Not Found

**Symptoms:**
```
FileNotFoundError: LibreOffice not found
Command 'libreoffice' not found
```

**Solution:**

**Linux/Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install libreoffice
```

**macOS:**
```bash
brew install --cask libreoffice
```

**Windows:**
- Download from [LibreOffice website](https://www.libreoffice.org/)
- Add to PATH or specify full path in configuration

### Problem: Permission Denied

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Fix file permissions
chmod +x src/generate_dataset.py

# Fix directory permissions
sudo chown -R $USER:$USER /path/to/XFUND_generator

# On Windows, run as Administrator
```

## Configuration Issues

### Problem: Path Does Not Exist

**Symptoms:**
```
ValidationError: Path does not exist: data/templates_docx
ValueError: CSV file not found: data/csv/data.csv
```

**Solution:**
```bash
# Check if paths exist
ls data/templates_docx/
ls data/csv/

# Create missing directories
mkdir -p data/templates_docx data/csv output

# Use absolute paths
config = GeneratorConfig(
    templates_dir="/full/path/to/templates",
    csv_path="/full/path/to/data.csv",
    output_dir="/full/path/to/output"
)
```

### Problem: Invalid Configuration

**Symptoms:**
```
ValidationError: 1 validation error for GeneratorConfig
```

**Solution:**
```python
# Validate configuration step by step
from src.models import GeneratorConfig, validate_config_file

# Check configuration file
result = validate_config_file("config.json")
if not result.is_valid:
    print("Errors:")
    for error in result.errors:
        print(f"  - {error}")

# Use default configuration as starting point
from src.models import get_default_config
config = get_default_config()
config.csv_path = "your_data.csv"  # Modify as needed
```

### Problem: Template Not Found

**Symptoms:**
```
No valid templates found
Template file does not exist
```

**Solution:**
```bash
# Check template directory structure
ls -la data/templates_docx/

# Should have both .docx and _layout.json files:
# medical_form.docx
# medical_form_layout.json

# Verify naming convention matches
python -c "
import os
templates_dir = 'data/templates_docx'
for f in os.listdir(templates_dir):
    print(f)
"
```

## Data Issues

### Problem: CSV Loading Error

**Symptoms:**
```
pandas.errors.EmptyDataError: No columns to parse from file
UnicodeDecodeError: 'utf-8' codec can't decode
```

**Solution:**
```bash
# Check CSV file format
head -n 5 data/csv/data.csv

# Ensure UTF-8 encoding
iconv -f ISO-8859-1 -t UTF-8 data.csv > data_utf8.csv

# Check for empty files
ls -la data/csv/data.csv
```

```python
# Validate CSV manually
import pandas as pd

try:
    df = pd.read_csv("data/csv/data.csv")
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
except Exception as e:
    print(f"CSV error: {e}")
```

### Problem: Data Validation Errors

**Symptoms:**
```
ValidationError: Entity text cannot be empty
ValueError: Invalid bbox coordinates
```

**Solution:**
```python
from src.models import DataRecord

# Check data records individually
import pandas as pd
df = pd.read_csv("data/csv/data.csv")

for i, row in df.iterrows():
    try:
        record = DataRecord(**row.to_dict())
        print(f"Row {i}: OK")
    except Exception as e:
        print(f"Row {i}: ERROR - {e}")
```

## Generation Issues

### Problem: No Images Generated

**Symptoms:**
- Empty output directory
- All entries marked as failed

**Solution:**
```python
# Enable debug mode
config.generate_debug_overlays = True

# Check generation result
result = generator.generate_dataset()
print(f"Generated: {result.generated_entries}")
print(f"Failed: {result.failed_entries}")

if result.errors:
    print("Errors:")
    for error in result.errors:
        print(f"  - {error}")
```

### Problem: Poor Quality Images

**Symptoms:**
- Blurry or distorted images
- Text not readable

**Solution:**
```python
# Adjust quality settings
config = GeneratorConfig(
    # ... other settings
    image_dpi=600,  # Higher resolution
    enable_augmentations=False,  # Disable for testing
    strict_validation=True,  # Enable quality checks
    generate_debug_overlays=True  # Check bounding boxes
)
```

### Problem: DOCX Conversion Fails

**Symptoms:**
```
Error converting DOCX to image
LibreOffice conversion failed
```

**Solution:**
```bash
# Test LibreOffice manually
libreoffice --version
libreoffice --headless --convert-to pdf test.docx

# Check file permissions
ls -la data/templates_docx/*.docx

# Try with different template
cp working_template.docx test_template.docx
```

## Performance Issues

### Problem: Slow Generation

**Symptoms:**
- Long processing times
- High memory usage

**Solution:**
```python
# Optimize configuration
config = GeneratorConfig(
    # ... other settings
    max_workers=2,  # Reduce workers
    batch_size=5,   # Smaller batches
    enable_augmentations=False  # Disable heavy processing
)

# Monitor resource usage
import psutil
print(f"CPU usage: {psutil.cpu_percent()}%")
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

### Problem: Memory Errors

**Symptoms:**
```
MemoryError: Unable to allocate array
Out of memory
```

**Solution:**
```python
# Reduce memory usage
config = GeneratorConfig(
    # ... other settings
    image_dpi=300,  # Lower resolution
    max_workers=1,  # Single worker
    batch_size=1    # Process one at a time
)

# Clear cache between generations
import gc
gc.collect()
```

## OCR Evaluation Issues

### Problem: OCR Results Not Loading

**Symptoms:**
```
FileNotFoundError: ocr_results.xlsx not found
Sheet not found in Excel file
```

**Solution:**
```bash
# Check file exists and format
ls -la ocr_results.xlsx
file ocr_results.xlsx

# Verify Excel sheet names
python -c "
import openpyxl
wb = openpyxl.load_workbook('ocr_results.xlsx')
print('Sheets:', wb.sheetnames)
"
```

### Problem: Evaluation Script Errors

**Symptoms:**
```
KeyError: Column not found
ValueError: Invalid metrics calculation
```

**Solution:**
```python
# Check data structure
import pandas as pd
df = pd.read_excel("ocr_results.xlsx", sheet_name="OCR Results")
print("Columns:", list(df.columns))
print("Shape:", df.shape)

# Use column mapping
python analyze_ocr_results.py --excel-file ocr_results.xlsx --debug
```

## Debugging Tips

### Enable Debug Mode

```python
# In configuration
config = GeneratorConfig(
    # ... other settings
    generate_debug_overlays=True,
    strict_validation=True
)

# In code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Intermediate Files

```bash
# Look for temporary files
ls -la /tmp/xfund_*

# Check generated images
ls -la output/images/

# Verify annotations
python -c "
import json
with open('output/annotations/0001.json') as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))
"
```

### Validate Step by Step

```python
# Test each component separately
from src.models import GeneratorConfig

# 1. Test configuration
config = GeneratorConfig.from_json_file("config.json")
print("✓ Config loaded")

# 2. Test data loading
from src.utils import load_csv_data_as_models
records = load_csv_data_as_models(config.csv_path)
print(f"✓ Loaded {len(records)} records")

# 3. Test template loading
generator = XFUNDGenerator(config)
templates = generator._find_templates()
print(f"✓ Found {len(templates)} templates")
```

## Common Error Patterns

### Pattern 1: Module Import Errors

**Cause:** Virtual environment not activated or missing dependencies

**Fix:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Pattern 2: Path Errors

**Cause:** Relative vs absolute paths, missing files

**Fix:**
```python
import os
print("Current directory:", os.getcwd())
print("Files:", os.listdir("."))

# Use absolute paths
config.templates_dir = os.path.abspath("data/templates_docx")
```

### Pattern 3: Data Format Errors

**Cause:** Mismatched CSV columns, encoding issues

**Fix:**
```python
# Check CSV structure
import pandas as pd
df = pd.read_csv("data.csv")
print("Columns:", df.columns.tolist())
print("Sample:", df.head(2))
```

## Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Enable debug mode and check logs**
3. **Test with minimal configuration**
4. **Verify installation and dependencies**

### Information to Include

When reporting issues, include:

- **Error message** (full traceback)
- **Configuration** (sanitized)
- **Environment** (OS, Python version)
- **Steps to reproduce**
- **Expected vs actual behavior**

### Where to Get Help

- **GitHub Issues**: [Report bugs and issues](https://github.com/danghoangnhan/XFUND_generator/issues)
- **GitHub Discussions**: [Ask questions and get help](https://github.com/danghoangnhan/XFUND_generator/discussions)
- **Documentation**: Check other guides in the `docs/` directory

## Emergency Fixes

### Reset Everything

```bash
# Clean virtual environment
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Test Installation

```bash
python demo_pydantic_integration.py
```

### Minimal Test

```python
from src.models import GeneratorConfig
config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/data.csv",
    output_dir="output/test"
)
print("✓ Configuration works")
```

---

*Still having issues? Check the other documentation files or open an issue on GitHub for help.*