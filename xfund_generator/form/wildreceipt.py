from typing import Any

from .base import BaseAnnotation, BaseDataset


# https://github.com/arkilpatra/WildReceipt
# ----------------------
# WildReceipt Annotation
# ----------------------
class WildReceiptAnnotation(BaseAnnotation):
    pass  # only box, text, label needed


class WildReceiptDataset(BaseDataset[WildReceiptAnnotation]):
    def _format_annotation_for_export(self, annotation: Any) -> dict[str, Any]:
        """Override to export only box, text, and label (minimal format)."""
        return {
            "box": annotation.box,
            "text": annotation.text,
            "label": annotation.label,
        }
