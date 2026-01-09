from enum import Enum
from typing import Optional

from pydantic import BaseModel


class LabelType(str, Enum):
    QUESTION = "question"
    ANSWER = "answer"
    OTHER = "other"


# ----------------------
# Word-level annotation
# ----------------------
class Word(BaseModel):
    box: list[int]
    text: str


# ----------------------
# Base annotation
# ----------------------
class BaseAnnotation(BaseModel):
    box: Optional[list[int]] = None
    text: str
    label: str
    words: Optional[list[Word]] = None
    id: Optional[int] = None


# ----------------------
# Base dataset
# ----------------------
class BaseDataset(BaseModel):
    image_path: str = ""
    annotations: list[BaseAnnotation]

    def _format_annotation_for_export(self, annotation: BaseAnnotation) -> dict:
        """
        Format a single annotation for JSON export.
        Override this method in subclasses to customize the export format.
        """
        return {
            "box": annotation.box,
            "text": annotation.text,
            "label": annotation.label,
            "words": [w.dict() for w in annotation.words] if annotation.words else [],
            "id": annotation.id,
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Unified JSON export method using OOP inheritance.
        Each subclass can customize the export format by overriding _format_annotation_for_export.
        """
        formatted_annotations = []
        for annotation in self.annotations:
            formatted_annotations.append(self._format_annotation_for_export(annotation))

        export_data = {"annotations": formatted_annotations}

        import json

        return json.dumps(export_data, indent=indent, ensure_ascii=False)
