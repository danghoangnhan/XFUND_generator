"""
Form annotation package for different document annotation formats.
Provides unified JSON export using OOP inheritance.
"""

from .base import BaseAnnotation, BaseDataset, LabelType, Word
from .funsd import FUNSDAnnotation, FUNSDDataset
from .wildreceipt import WildReceiptAnnotation, WildReceiptDataset
from .xfund import XFUNDAnnotation, XFUNDDataset

__all__ = [
    "BaseAnnotation",
    "BaseDataset",
    "LabelType",
    "Word",
    "XFUNDAnnotation",
    "XFUNDDataset",
    "FUNSDAnnotation",
    "FUNSDDataset",
    "WildReceiptAnnotation",
    "WildReceiptDataset",
]
