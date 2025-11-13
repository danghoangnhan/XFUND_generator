from typing import List, Dict, Tuple
from pydantic import field_validator, model_validator
from .base import BaseDataset, BaseAnnotation, Word, LabelType

#"https://arxiv.org/abs/2003.13353"
class XFUNDAnnotation(BaseAnnotation):
    linking: List[List[int]] = []

class XFUNDDataset(BaseDataset):
    annotations: List[XFUNDAnnotation]

    question_to_answer_ids: Dict[int, List[int]] = {}
    question_to_answer_text: Dict[str, List[str]] = {}

    @model_validator(mode='after')
    def build_mappings(self):
        """Build question → answer mappings after model validation."""
        # Build question → answer id mapping
        mapping_ids = {}
        for ann in self.annotations:
            if ann.label == "question" and ann.linking:
                mapping_ids[ann.id] = [link[1] for link in ann.linking]
        self.question_to_answer_ids = mapping_ids
        
        # Build question → answer text mapping
        id_to_text = {ann.id: ann.text for ann in self.annotations if ann.id}
        mapping_text = {}
        for qid, a_ids in mapping_ids.items():
            if qid in id_to_text:
                mapping_text[id_to_text[qid]] = [id_to_text[aid] for aid in a_ids if aid in id_to_text]
        self.question_to_answer_text = mapping_text
        
        return self

    def get_grouped_qa_pairs(self) -> List[Tuple[str, List[str]]]:
        return list(self.question_to_answer_text.items())

    def get_flat_qa_pairs(self) -> List[Tuple[str, str]]:
        flat_pairs = []
        for q, a_list in self.question_to_answer_text.items():
            for a in a_list:
                flat_pairs.append((q, a))
        return flat_pairs

    def _format_annotation_for_export(self, annotation: 'XFUNDAnnotation') -> dict:
        """Override to include XFUND-specific linking information."""
        base_format = super()._format_annotation_for_export(annotation)
        base_format["linking"] = annotation.linking
        return base_format
