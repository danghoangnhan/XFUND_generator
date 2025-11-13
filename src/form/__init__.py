"""
Form annotation package for different document annotation formats.
Provides unified JSON export using OOP inheritance.
"""

from .base import BaseAnnotation, BaseDataset, Word
from .xfund import XFUNDAnnotation, XFUNDDataset
from .funsd import FUNSDAnnotation, FUNSDDataset
from .wildreceipt import WildReceiptAnnotation, WildReceiptDataset

__all__ = [
    'BaseAnnotation', 'BaseDataset', 'Word',
    'XFUNDAnnotation', 'XFUNDDataset',
    'FUNSDAnnotation', 'FUNSDDataset',
    'WildReceiptAnnotation', 'WildReceiptDataset'
]