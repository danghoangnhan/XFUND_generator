# Testing Guide

Comprehensive guide to testing XFUND Generator.

## Quick Start

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Quick tests only
uv run pytest -m "not slow" -v
```

## Test Commands

### Make Commands

| Command | Description |
|---------|-------------|
| `make test` | Run all tests |
| `make test-cov` | Run with coverage report |
| `make lint` | Check code style |
| `make format` | Format code |
| `make type-check` | Run type checking |

### Direct pytest

```bash
# All tests
uv run pytest tests/ -v

# By marker
uv run pytest -m "unit" -v           # Unit tests
uv run pytest -m "integration" -v    # Integration tests
uv run pytest -m "pydantic" -v       # Pydantic tests
uv run pytest -m "forms" -v          # Form class tests
uv run pytest -m "not slow" -v       # Exclude slow tests

# Specific files
uv run pytest tests/test_pydantic_models.py -v
uv run pytest tests/test_form_classes.py -v
uv run pytest tests/test_generator.py -v
uv run pytest tests/test_integration.py -v

# With coverage
uv run pytest --cov=xfund_generator --cov-report=html tests/
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | Fast unit tests |
| `@pytest.mark.integration` | End-to-end tests |
| `@pytest.mark.pydantic` | Pydantic model tests |
| `@pytest.mark.forms` | Form class tests |
| `@pytest.mark.config` | Configuration tests |
| `@pytest.mark.slow` | Long-running tests |

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_pydantic_models.py  # Pydantic model tests
├── test_form_classes.py     # Form class tests
├── test_generator.py        # Generator tests
└── test_integration.py      # Integration tests
```

## Test Coverage Areas

### Pydantic Models (`test_pydantic_models.py`)

- BBoxModel validation and computed properties
- GeneratorConfig validation and path resolution
- DataRecord and TemplateValidationResult
- Model serialization/deserialization

```python
@pytest.mark.pydantic
def test_bbox_validation():
    bbox = BBoxModel(x1=10, y1=20, x2=100, y2=80)
    assert bbox.width == 90
    assert bbox.height == 60
```

### Form Classes (`test_form_classes.py`)

- Base classes (Word, BaseAnnotation, BaseDataset)
- XFUND, FUNSD, WildReceipt format classes
- Unified JSON export API
- Template Method pattern implementation

```python
@pytest.mark.forms
def test_xfund_annotation():
    annotation = XFUNDAnnotation(
        id=0,
        text="Test",
        box=[10, 20, 100, 80],
        label="question",
        words=[],
        linking=[]
    )
    assert annotation.label == "question"
```

### Integration (`test_integration.py`)

- XFUND form generator functionality
- Question-answer linking automation
- Word-level annotation creation
- Complete pipeline testing

```python
@pytest.mark.integration
def test_full_pipeline():
    config = GeneratorConfig(...)
    generator = XFUNDGenerator(config)
    result = generator.generate_dataset()
    assert result.generated_entries > 0
```

### Generator Core (`test_generator.py`)

- DOCX processing utilities
- Document rendering
- Image augmentation features

## Debugging Tests

```bash
# Run only failed tests
uv run pytest --lf -v

# Debug mode with detailed output
uv run pytest -v -s --tb=long --showlocals

# Stop on first failure
uv run pytest -x -v

# Pattern matching
uv run pytest -k "test_bbox" -v
```

## Writing Tests

### Adding New Tests

1. Choose appropriate test file based on functionality
2. Add relevant markers (`@pytest.mark.unit`, etc.)
3. Use existing fixtures from `conftest.py`
4. Follow naming: `test_feature_description`

### Example Test

```python
import pytest
from xfund_generator import BBoxModel

@pytest.mark.unit
@pytest.mark.pydantic
def test_bbox_center_calculation():
    """Test that bbox center is calculated correctly."""
    bbox = BBoxModel(x1=0, y1=0, x2=100, y2=100)
    center = bbox.center
    assert center == (50.0, 50.0)
```

### Using Fixtures

```python
# In conftest.py
@pytest.fixture
def sample_config():
    return GeneratorConfig(
        templates_dir="data/templates_docx",
        csv_path="data/csv/data.csv",
        output_dir="output"
    )

# In test file
def test_generator_creation(sample_config):
    generator = XFUNDGenerator(sample_config)
    assert generator is not None
```

## CI/CD

GitHub Actions runs tests on:
- Push to `master`/`main`
- Pull requests
- Tag pushes (`v*`) for releases

**Tested Python versions:** 3.9, 3.10, 3.11, 3.12, 3.13, 3.14

### CI Configuration

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    uv run pytest tests/ -v --cov=xfund_generator
```

## Coverage

Generate coverage reports:

```bash
# HTML report
uv run pytest --cov=xfund_generator --cov-report=html tests/

# Terminal report
uv run pytest --cov=xfund_generator --cov-report=term-missing tests/
```

View HTML report: `open htmlcov/index.html`

## See Also

- [[Getting-Started]] - Basic usage
- [[API-Reference]] - API documentation
- [[Contributing]] - Contribution guidelines
