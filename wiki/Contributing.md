# Contributing

Guidelines for contributing to XFUND Generator.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/XFUND_generator.git
   cd XFUND_generator
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Code refactoring

### Make Changes

1. Write code following the style guide
2. Add tests for new functionality
3. Update documentation if needed

### Run Quality Checks

```bash
# Format code
make format

# Check style
make lint

# Run type checking
make type-check

# Run tests
make test
```

### Commit Changes

```bash
git add .
git commit -m "Add feature: description"
```

Commit message format:
- `Add` - New features
- `Fix` - Bug fixes
- `Update` - Changes to existing features
- `Remove` - Removed features
- `Refactor` - Code refactoring
- `Docs` - Documentation changes

### Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style

### Python Style

- Follow PEP 8
- Use type hints
- Maximum line length: 88 characters (Black default)
- Use meaningful variable names

### Formatting

Code is formatted with `ruff`:

```bash
make format
```

### Linting

```bash
make lint
```

### Type Checking

```bash
make type-check
```

## Testing

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific markers
uv run pytest -m "unit" -v
```

### Writing Tests

1. Place tests in `tests/` directory
2. Use appropriate markers (`@pytest.mark.unit`, etc.)
3. Use fixtures from `conftest.py`
4. Follow naming: `test_feature_description`

Example:

```python
import pytest
from xfund_generator import BBoxModel

@pytest.mark.unit
@pytest.mark.pydantic
def test_bbox_validation():
    bbox = BBoxModel(x1=10, y1=20, x2=100, y2=80)
    assert bbox.width == 90
```

## Pull Request Guidelines

### Before Submitting

- [ ] All tests pass
- [ ] Code is formatted
- [ ] Linting passes
- [ ] Type checking passes
- [ ] Documentation updated (if needed)

### PR Description

Include:
- Summary of changes
- Related issues (if any)
- Test coverage
- Breaking changes (if any)

### Review Process

1. Maintainers review the PR
2. Address feedback
3. PR is merged after approval

## Reporting Issues

### Bug Reports

Include:
- Python version
- OS and version
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/logs

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternatives considered

## Documentation

### Updating Docs

- Local docs: `docs/` directory
- Wiki: `wiki/` directory

### Wiki Updates

```bash
# Clone wiki
git clone https://github.com/danghoangnhan/XFUND_generator.wiki.git

# Copy updated files from wiki/ to the cloned wiki repo
# Commit and push
```

## Release Process

Releases are automated via GitHub Actions when tags are pushed:

```bash
git tag v1.0.1
git push origin v1.0.1
```

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn

## Questions?

- Open an [issue](https://github.com/danghoangnhan/XFUND_generator/issues)
- Check existing issues and wiki first
