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
__author__ = "Tu Hung Jen"

# Core models and configuration
from .models import (
    GeneratorConfig,
    BBoxModel,
    DataRecord,
    XFUNDEntity,
    TemplateValidationResult,
    DocumentType,
    AugmentationDifficulty
)

# Form classes and base functionality
from .form import (
    BaseDataset,
    XFUNDDataset,
    FUNSDDataset,
    WildReceiptDataset,
    Word,
    XFUNDAnnotation,
    FUNSDAnnotation,
    WildReceiptAnnotation
)

# Core functionality
from .generate_dataset import XFUNDGenerator
from .utils import validate_docx_template, load_csv_data
from .renderer import DocumentRenderer
from .augmentations import DocumentAugmenter

# Integration modules
from .xfund_form_integration import XFUNDFormGenerator

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Core models
    "GeneratorConfig",
    "BBoxModel", 
    "DataRecord",
    "XFUNDEntity",
    "TemplateValidationResult",
    "DocumentType",
    "AugmentationDifficulty",
    
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
    "DocumentRenderer",
    "DocumentAugmenter",
    "XFUNDFormGenerator",
    
    # Utilities
    "validate_docx_template",
    "load_csv_data",
]