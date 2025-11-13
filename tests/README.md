# Test Suite

This directory contains comprehensive test cases for the XFUND Generator project using pytest.

## Test Structure

### Core Test Files

- `test_pydantic_models.py` - Tests for Pydantic model validation and functionality
- `test_form_classes.py` - Tests for form classes and OOP inheritance
- `test_integration.py` - Integration tests for XFUND form generation
- `test_generator.py` - Tests for core generator functionality
- `conftest.py` - Shared pytest fixtures and configuration

### Support Files

- `pytest.ini` - Pytest configuration and markers
- `old_demos/` - Backup of original demo scripts

## Running Tests

### Run All Tests
```bash
# From project root
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage
pytest tests/ --cov=src
```

### Run Specific Test Categories
```bash
# Unit tests only (fast)
pytest tests/ -m unit

# Integration tests only
pytest tests/ -m integration

# Pydantic-related tests
pytest tests/ -m pydantic

# Form classes tests
pytest tests/ -m forms

# Configuration tests
pytest tests/ -m config
```

### Run Specific Test Files
```bash
# Test Pydantic models
pytest tests/test_pydantic_models.py

# Test form classes
pytest tests/test_form_classes.py

# Test integration
pytest tests/test_integration.py

# Test generator core
pytest tests/test_generator.py
```

### Exclude Slow Tests
```bash
# Skip slow tests
pytest tests/ -m "not slow"
```

## Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests (slower)
- `@pytest.mark.pydantic` - Pydantic validation tests
- `@pytest.mark.forms` - Form classes tests
- `@pytest.mark.config` - Configuration tests
- `@pytest.mark.slow` - Slow tests that can be skipped

## Test Coverage

The test suite covers:

### Pydantic Models (`test_pydantic_models.py`)
- ✅ BBoxModel validation and computed properties
- ✅ GeneratorConfig validation and path resolution
- ✅ DataRecord validation
- ✅ TemplateValidationResult functionality
- ✅ XFUNDEntity validation
- ✅ Model serialization/deserialization
- ✅ Configuration file validation

### Form Classes (`test_form_classes.py`)
- ✅ Base classes (Word, BaseAnnotation, BaseDataset)
- ✅ XFUND classes and question-answer mappings
- ✅ FUNSD classes and key/value relationships
- ✅ WildReceipt classes (minimal format)
- ✅ Unified JSON export API across all formats
- ✅ Polymorphic behavior and inheritance benefits
- ✅ Template Method pattern implementation
- ✅ Extensibility for new formats

### Integration Tests (`test_integration.py`)
- ✅ XFUND form generator functionality
- ✅ Question-answer linking automation
- ✅ Word-level annotation creation
- ✅ Medical field detection and classification
- ✅ Unified JSON export consistency
- ✅ Large dataset handling
- ✅ Complete pipeline testing
- ✅ Error handling and validation
- ✅ Backwards compatibility

### Generator Core (`test_generator.py`)
- ✅ Utility functions (when available)
- ✅ DOCX processing utilities
- ✅ Word rendering functionality
- ✅ Image augmentation features
- ✅ Integration with modern config
- ✅ Validation functions

## Fixtures

The `conftest.py` file provides shared fixtures:

- `temp_dir` - Temporary directory for test files
- `sample_config_data` - Sample configuration dictionary
- `sample_config` - GeneratorConfig instance
- `sample_bbox` - BBoxModel instance
- `sample_word` - Word instance
- `sample_*_annotation` - Annotation instances for each format
- `sample_datasets` - Complete dataset instances for all formats
- `mock_file_operations` - Mocked file I/O operations

## Architecture Benefits Tested

The test suite validates the architectural improvements:

### OOP Inheritance Benefits
- ✅ No redundant format-specific methods
- ✅ Unified API across all formats
- ✅ Template Method pattern implementation
- ✅ Polymorphic behavior
- ✅ Easy extensibility for new formats

### Pydantic Integration Benefits
- ✅ Type safety and validation
- ✅ Better error messages
- ✅ Model serialization/deserialization
- ✅ Configuration validation
- ✅ Path handling and resolution

## Adding New Tests

### For New Features
1. Choose appropriate test file based on functionality
2. Add relevant markers (`@pytest.mark.unit`, etc.)
3. Use existing fixtures when possible
4. Follow naming convention: `test_feature_description`

### For New Form Formats
1. Add fixtures to `conftest.py`
2. Add test cases to `test_form_classes.py`
3. Ensure unified API compatibility
4. Test format-specific features

### For Integration Features
1. Add to `test_integration.py`
2. Use `@pytest.mark.integration` marker
3. Mock external dependencies
4. Test complete workflows

## Continuous Integration

The test suite is designed for CI/CD:

- Fast unit tests for quick feedback
- Comprehensive integration tests for full validation
- Proper mocking to avoid external dependencies
- Clear markers for selective test execution
- Good error messages for debugging failures

## Migration Notes

The test suite replaces the previous demo-based testing approach:

- **Before**: Individual demo scripts with manual verification
- **After**: Automated pytest suite with assertions
- **Benefits**: Faster feedback, better coverage, CI/CD integration
- **Backwards Compatibility**: Old demos preserved in `old_demos/`