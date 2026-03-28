"""
XFUND Generator Package

A comprehensive toolkit for generating XFUND-style OCR datasets with document templates,
automatic annotation, and support for multiple annotation formats (XFUND, FUNSD, WildReceipt).

Features:
- Template-based document generation from DOCX files
- Multiple annotation formats with unified API
- Pydantic v2 integration for type safety
- Advanced augmentations and quality validation
- OCR evaluation tools
"""

__version__ = "1.0.0"
__author__ = "Daniel Tu"

from .augmentations import DocumentAugmenter
from .docx_utils import validate_docx_template

# Form classes and base functionality
from .form import (
    BaseDataset,
    FUNSDAnnotation,
    FUNSDDataset,
    WildReceiptAnnotation,
    WildReceiptDataset,
    Word,
    XFUNDAnnotation,
    XFUNDDataset,
)

# Core functionality
from .generate_dataset import XFUNDGenerator

# Core models and configuration
from .models import (
    AugmentationConfig,
    AugmentationDifficulty,
    AugmentationQualityResult,
    BatchEntryResult,
    BBoxModel,
    DataRecord,
    EntryResult,
    GeneratorConfig,
    LayoutConfig,
    LayoutField,
    SetupValidationResult,
    TemplateValidationResult,
    XFUNDEntity,
)
from .renderer import WordRenderer
from .utils import load_csv_data

# Integration modules
from .xfund_form_integration import XFUNDFormGenerator

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core models
    "AugmentationConfig",
    "AugmentationDifficulty",
    "AugmentationQualityResult",
    "BatchEntryResult",
    "BBoxModel",
    "DataRecord",
    "EntryResult",
    "GeneratorConfig",
    "LayoutConfig",
    "LayoutField",
    "SetupValidationResult",
    "TemplateValidationResult",
    "XFUNDEntity",
    # Form classes
    "BaseDataset",
    "XFUNDDataset",
    "FUNSDDataset",
    "WildReceiptDataset",
    "Word",
    "XFUNDAnnotation",
    "FUNSDAnnotation",
    "WildReceiptAnnotation",
    # Core functionality
    "XFUNDGenerator",
    "WordRenderer",
    "DocumentAugmenter",
    "XFUNDFormGenerator",
    # Utilities
    "validate_docx_template",
    "load_csv_data",
]
