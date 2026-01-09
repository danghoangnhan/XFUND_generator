# Annotation Formats

XFUND Generator supports three annotation formats with a unified API.

## Unified API

All formats use the same interface:

```python
from xfund_generator.form.xfund import XFUNDDataset
from xfund_generator.form.funsd import FUNSDDataset
from xfund_generator.form.wildreceipt import WildReceiptDataset

# All formats work the same way
dataset = XFUNDDataset(image_path="image.png")
dataset.add_annotation(...)
json_output = dataset.to_json()
```

## XFUND Format

XFUND format includes entity linking information for key-value relationships.

### Structure

```json
{
  "annotations": [
    {
      "id": 0,
      "text": "Patient Name:",
      "box": [50, 100, 150, 120],
      "label": "question",
      "words": [
        {"text": "Patient", "box": [50, 100, 100, 120]},
        {"text": "Name:", "box": [105, 100, 150, 120]}
      ],
      "linking": [[0, 1]]
    },
    {
      "id": 1,
      "text": "John Smith",
      "box": [160, 100, 250, 120],
      "label": "answer",
      "words": [
        {"text": "John", "box": [160, 100, 190, 120]},
        {"text": "Smith", "box": [195, 100, 250, 120]}
      ],
      "linking": []
    }
  ]
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique annotation ID |
| `text` | string | Full text content |
| `box` | array | Bounding box [x1, y1, x2, y2] |
| `label` | string | Label: question, answer, header, other |
| `words` | array | Word-level annotations |
| `linking` | array | Links to related annotations [[from_id, to_id], ...] |

### Usage

```python
from xfund_generator.form.xfund import XFUNDDataset, XFUNDAnnotation
from xfund_generator.form.base import Word

dataset = XFUNDDataset(image_path="doc.png")

# Question annotation
question = XFUNDAnnotation(
    id=0,
    text="Patient Name:",
    box=[50, 100, 150, 120],
    label="question",
    words=[
        Word(text="Patient", box=[50, 100, 100, 120]),
        Word(text="Name:", box=[105, 100, 150, 120])
    ],
    linking=[[0, 1]]  # Links to answer (id=1)
)

# Answer annotation
answer = XFUNDAnnotation(
    id=1,
    text="John Smith",
    box=[160, 100, 250, 120],
    label="answer",
    words=[
        Word(text="John", box=[160, 100, 190, 120]),
        Word(text="Smith", box=[195, 100, 250, 120])
    ],
    linking=[]
)

dataset.add_annotation(question)
dataset.add_annotation(answer)
output = dataset.to_json()
```

## FUNSD Format

FUNSD format uses explicit key_id/value_id for relationships.

### Structure

```json
{
  "annotations": [
    {
      "id": 0,
      "text": "Patient Name:",
      "box": [50, 100, 150, 120],
      "label": "question",
      "words": [
        {"text": "Patient", "box": [50, 100, 100, 120]},
        {"text": "Name:", "box": [105, 100, 150, 120]}
      ],
      "key_id": 0,
      "value_id": 1
    }
  ]
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique annotation ID |
| `text` | string | Full text content |
| `box` | array | Bounding box [x1, y1, x2, y2] |
| `label` | string | Label: question, answer, header, other |
| `words` | array | Word-level annotations |
| `key_id` | integer | ID of the key annotation |
| `value_id` | integer | ID of the value annotation |

### Usage

```python
from xfund_generator.form.funsd import FUNSDDataset, FUNSDAnnotation
from xfund_generator.form.base import Word

dataset = FUNSDDataset(image_path="doc.png")

annotation = FUNSDAnnotation(
    id=0,
    text="Patient Name:",
    box=[50, 100, 150, 120],
    label="question",
    words=[Word(text="Patient", box=[50, 100, 100, 120])],
    key_id=0,
    value_id=1
)

dataset.add_annotation(annotation)
output = dataset.to_json()
```

## WildReceipt Format

WildReceipt is a minimal format without relationship information.

### Structure

```json
{
  "annotations": [
    {
      "id": 0,
      "text": "Patient Name:",
      "box": [50, 100, 150, 120],
      "label": "question",
      "words": [
        {"text": "Patient", "box": [50, 100, 100, 120]},
        {"text": "Name:", "box": [105, 100, 150, 120]}
      ]
    }
  ]
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique annotation ID |
| `text` | string | Full text content |
| `box` | array | Bounding box [x1, y1, x2, y2] |
| `label` | string | Label category |
| `words` | array | Word-level annotations |

### Usage

```python
from xfund_generator.form.wildreceipt import WildReceiptDataset, WildReceiptAnnotation
from xfund_generator.form.base import Word

dataset = WildReceiptDataset(image_path="doc.png")

annotation = WildReceiptAnnotation(
    id=0,
    text="Total: $100",
    box=[50, 100, 150, 120],
    label="total",
    words=[
        Word(text="Total:", box=[50, 100, 90, 120]),
        Word(text="$100", box=[95, 100, 150, 120])
    ]
)

dataset.add_annotation(annotation)
output = dataset.to_json()
```

## Format Comparison

| Feature | XFUND | FUNSD | WildReceipt |
|---------|-------|-------|-------------|
| Word-level boxes | Yes | Yes | Yes |
| Entity linking | Yes (linking array) | Yes (key_id/value_id) | No |
| Use case | Document understanding | Form understanding | Receipt extraction |

## Polymorphic Usage

```python
def process_dataset(dataset):
    """Works with any format"""
    return dataset.to_json()

# Use with any format
xfund_output = process_dataset(XFUNDDataset("doc.png"))
funsd_output = process_dataset(FUNSDDataset("doc.png"))
wild_output = process_dataset(WildReceiptDataset("doc.png"))
```

## See Also

- [[Getting-Started]] - Basic usage
- [[API-Reference]] - Full API documentation
