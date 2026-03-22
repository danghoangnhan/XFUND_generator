"""
Legacy form models - DEPRECATED.

This module is deprecated. Use the `xfund_generator.form` package instead:
    - xfund_generator.form.base (BaseAnnotation, BaseDataset, Word)
    - xfund_generator.form.xfund (XFUNDAnnotation, XFUNDDataset)
    - xfund_generator.form.funsd (FUNSDAnnotation, FUNSDDataset)
    - xfund_generator.form.wildreceipt (WildReceiptAnnotation, WildReceiptDataset)

This file is kept only for backward compatibility and will be removed in v2.0.
"""

import warnings

from pydantic import BaseModel, model_validator

warnings.warn(
    "xfund_generator.form (form.py) is deprecated. "
    "Use xfund_generator.form.xfund, .funsd, or .wildreceipt instead.",
    DeprecationWarning,
    stacklevel=2,
)


# Word-level annotation
class Word(BaseModel):
    box: list[int]  # [x0, y0, x1, y1]
    text: str


# Main annotation (question or answer)
class Annotation(BaseModel):
    box: list[int]
    text: str
    label: str  # "question" or "answer"
    words: list[Word]
    linking: list[list[int]]  # list of [start_id, end_id]
    id: int


# XFUND dataset wrapper
class XFUNDData(BaseModel):
    annotations: list[Annotation]

    # Auto-generated mappings
    question_to_answer_ids: dict[int, list[int]] = {}
    question_to_answer_text: dict[str, list[str]] = {}

    @model_validator(mode="after")
    def build_mappings(self) -> "XFUNDData":
        """Build question-to-answer mappings from annotations."""
        id_mapping: dict[int, list[int]] = {}
        id_to_text: dict[int, str] = {}

        for ann in self.annotations:
            id_to_text[ann.id] = ann.text
            if ann.label == "question":
                linked_ids = [link[1] for link in ann.linking]
                id_mapping[ann.id] = linked_ids

        self.question_to_answer_ids = id_mapping

        text_mapping: dict[str, list[str]] = {}
        for qid, a_ids in id_mapping.items():
            q_text = id_to_text.get(qid, "")
            text_mapping[q_text] = [
                id_to_text.get(aid, "") for aid in a_ids
            ]
        self.question_to_answer_text = text_mapping

        return self

    # ----------------------
    # Flattened list of QA pairs
    # ----------------------
    def get_flat_qa_pairs(self) -> list[tuple[str, str]]:
        flat_pairs = []
        for q_text, a_texts in self.question_to_answer_text.items():
            for a_text in a_texts:
                flat_pairs.append((q_text, a_text))
        return flat_pairs
