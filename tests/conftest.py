"""
Pytest configuration and shared fixtures.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from xfund_generator.form import (
    FUNSDAnnotation,
    FUNSDDataset,
    WildReceiptAnnotation,
    WildReceiptDataset,
    Word,
    XFUNDAnnotation,
    XFUNDDataset,
)
from xfund_generator.models import BBoxModel, GeneratorConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {
        "templates_dir": "data/templates_docx",
        "csv_path": "data/csv/data.csv",
        "output_dir": "output/test",
        "document_type": "medical",
        "enable_augmentations": True,
        "augmentation_difficulty": "medium",
        "image_dpi": 300,
        "target_size": 1000,
    }


@pytest.fixture
def sample_config(sample_config_data):
    """Sample GeneratorConfig instance for testing."""
    return GeneratorConfig(**sample_config_data)


@pytest.fixture
def sample_bbox():
    """Sample BBoxModel for testing."""
    return BBoxModel(x1=10, y1=20, x2=100, y2=80)


@pytest.fixture
def sample_word():
    """Sample Word for testing."""
    return Word(box=[10, 20, 50, 40], text="Sample")


@pytest.fixture
def sample_xfund_annotation(sample_word):
    """Sample XFUNDAnnotation for testing."""
    return XFUNDAnnotation(
        id=1,
        box=[10, 20, 100, 80],
        text="Patient Name:",
        label="question",
        words=[sample_word],
        linking=[[1, 2]],
    )


@pytest.fixture
def sample_funsd_annotation(sample_word):
    """Sample FUNSDAnnotation for testing."""
    return FUNSDAnnotation(
        id=1,
        box=[10, 20, 100, 80],
        text="Name:",
        label="question",
        words=[sample_word],
        key_id=1,
        value_id=2,
    )


@pytest.fixture
def sample_wildreceipt_annotation(sample_word):
    """Sample WildReceiptAnnotation for testing."""
    return WildReceiptAnnotation(
        id=1,
        box=[10, 20, 100, 80],
        text="Total:",
        label="question",
        words=[sample_word],
    )


@pytest.fixture
def sample_datasets(
    sample_xfund_annotation, sample_funsd_annotation, sample_wildreceipt_annotation
):
    """Sample datasets for all formats."""
    return {
        "xfund": XFUNDDataset(
            image_path="test.png", annotations=[sample_xfund_annotation]
        ),
        "funsd": FUNSDDataset(
            image_path="test.png", annotations=[sample_funsd_annotation]
        ),
        "wildreceipt": WildReceiptDataset(
            image_path="test.png", annotations=[sample_wildreceipt_annotation]
        ),
    }


@pytest.fixture
def mock_file_operations(monkeypatch):
    """Mock file operations for testing without actual file I/O."""
    mock_open = MagicMock()
    mock_exists = MagicMock(return_value=True)
    mock_isdir = MagicMock(return_value=True)
    mock_listdir = MagicMock(return_value=["template1.docx", "template2.docx"])

    monkeypatch.setattr("builtins.open", mock_open)
    monkeypatch.setattr("os.path.exists", mock_exists)
    monkeypatch.setattr("os.path.isdir", mock_isdir)
    monkeypatch.setattr("os.listdir", mock_listdir)

    return {
        "open": mock_open,
        "exists": mock_exists,
        "isdir": mock_isdir,
        "listdir": mock_listdir,
    }
