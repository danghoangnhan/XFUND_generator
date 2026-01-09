from typing import Optional

from .base import BaseAnnotation, BaseDataset


# ----------------------
# FUNSD Annotation
# ----------------------
class FUNSDAnnotation(BaseAnnotation):
    key_id: Optional[int] = None
    value_id: Optional[int] = None


# ----------------------
# FUNSD Dataset
# ----------------------
class FUNSDDataset(BaseDataset):
    annotations: list[FUNSDAnnotation]

    def _format_annotation_for_export(self, annotation: "FUNSDAnnotation") -> dict:
        """Override to include FUNSD-specific key_id and value_id information."""
        base_format = super()._format_annotation_for_export(annotation)
        base_format["key_id"] = annotation.key_id
        base_format["value_id"] = annotation.value_id
        return base_format
