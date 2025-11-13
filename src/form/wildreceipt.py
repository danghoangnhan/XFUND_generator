from typing import List
from .base import BaseDataset, BaseAnnotation, LabelType
from pydantic import BaseModel

#https://github.com/arkilpatra/WildReceipt
# ----------------------
# WildReceipt Annotation
# ----------------------
class WildReceiptAnnotation(BaseAnnotation):
    pass  # only box, text, label needed

class WildReceiptDataset(BaseDataset):
    annotations: List[WildReceiptAnnotation]

    def _format_annotation_for_export(self, annotation: 'WildReceiptAnnotation') -> dict:
        """Override to export only box, text, and label (minimal format)."""
        return {
            "box": annotation.box,
            "text": annotation.text,
            "label": annotation.label
        }
