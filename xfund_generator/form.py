from pydantic import BaseModel, validator


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

    @validator("question_to_answer_ids", pre=True, always=True)
    def build_question_to_answer_ids(cls, _v, values):
        mapping = {}
        annotations = values.get("annotations", [])
        for ann in annotations:
            if ann.label == "question":
                linked_ids = [link[1] for link in ann.linking]
                mapping[ann.id] = linked_ids
        return mapping

    @validator("question_to_answer_text", pre=True, always=True)
    def build_question_to_answer_text(cls, _v, values):
        annotations = values.get("annotations", [])
        id_to_text = {ann.id: ann.text for ann in annotations}
        q_to_a_ids = values.get("question_to_answer_ids", {})
        mapping = {}
        for qid, a_ids in q_to_a_ids.items():
            mapping[id_to_text[qid]] = [id_to_text[aid] for aid in a_ids]
        return mapping

    # ----------------------
    # Flattened list of QA pairs
    # ----------------------
    def get_flat_qa_pairs(self) -> list[tuple[str, str]]:
        flat_pairs = []
        for q_text, a_texts in self.question_to_answer_text.items():
            for a_text in a_texts:
                flat_pairs.append((q_text, a_text))
        return flat_pairs


# ----------------------
# Example usage
# ----------------------
annotations = [
    Annotation(
        box=[1794, 1748, 1985, 1823],
        text="实施周期",
        label="question",
        words=[
            Word(box=[1838, 1787, 1865, 1818], text="实"),
            Word(box=[1865, 1786, 1892, 1819], text="施"),
            Word(box=[1889, 1787, 1915, 1816], text="周"),
            Word(box=[1913, 1785, 1944, 1816], text="期"),
        ],
        linking=[[1104, 1128]],
        id=1104,
    ),
    Annotation(
        box=[907, 1836, 1091, 1873],
        text="普通高职在校生",
        label="answer",
        words=[
            Word(box=[908, 1837, 921, 1871], text="普"),
            Word(box=[927, 1837, 948, 1872], text="通"),
            Word(box=[954, 1837, 975, 1872], text="高"),
            Word(box=[981, 1837, 1002, 1873], text="职"),
            Word(box=[1007, 1838, 1026, 1873], text="在"),
            Word(box=[1032, 1838, 1053, 1873], text="校"),
            Word(box=[1059, 1838, 1084, 1874], text="生"),
        ],
        linking=[],
        id=1128,
    ),
]

xfund_data = XFUNDData(annotations=annotations)

print("Flat QA pairs:")
for q, a in xfund_data.get_flat_qa_pairs():
    print(f"Q: {q} → A: {a}")
