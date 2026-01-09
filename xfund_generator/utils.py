"""
Utility functions for XFUND dataset generation.
Provides bbox normalization, font selection, file handling, and other helpers.
Enhanced with Pydantic support for better type safety and validation.
"""

import glob
import json
import os
import random
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageFont

from .models import BBoxModel, DataRecord, GeneratorConfig, XFUNDAnnotation, XFUNDEntity


# Legacy BBox class for backward compatibility
class BBox:
    """Legacy bounding box utility class - use BBoxModel for new code."""

    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def to_list(self) -> list[float]:
        """Convert to list format [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]

    def to_xfund_format(self) -> list[int]:
        """Convert to XFUND integer format [x1, y1, x2, y2]."""
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]

    def to_pydantic(self) -> BBoxModel:
        """Convert legacy BBox to Pydantic BBoxModel."""
        return BBoxModel(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    def normalize(
        self, img_width: int, img_height: int, target_size: int = 1000
    ) -> "BBox":
        """Normalize bbox to 0-target_size scale (default XFUND 0-1000)."""
        norm_x1 = (self.x1 / img_width) * target_size
        norm_y1 = (self.y1 / img_height) * target_size
        norm_x2 = (self.x2 / img_width) * target_size
        norm_y2 = (self.y2 / img_height) * target_size
        return BBox(norm_x1, norm_y1, norm_x2, norm_y2)

    def denormalize(
        self, img_width: int, img_height: int, source_size: int = 1000
    ) -> "BBox":
        """Denormalize bbox from 0-source_size scale to actual image coordinates."""
        actual_x1 = (self.x1 / source_size) * img_width
        actual_y1 = (self.y1 / source_size) * img_height
        actual_x2 = (self.x2 / source_size) * img_width
        actual_y2 = (self.y2 / source_size) * img_height
        return BBox(actual_x1, actual_y1, actual_x2, actual_y2)

    def add_jitter(self, max_jitter: int = 5) -> "BBox":
        """Add small random jitter to bbox coordinates for realism."""
        jitter_x1 = random.randint(-max_jitter, max_jitter)
        jitter_y1 = random.randint(-max_jitter, max_jitter)
        jitter_x2 = random.randint(-max_jitter, max_jitter)
        jitter_y2 = random.randint(-max_jitter, max_jitter)

        return BBox(
            max(0, self.x1 + jitter_x1),
            max(0, self.y1 + jitter_y1),
            self.x2 + jitter_x2,
            self.y2 + jitter_y2,
        )

    def width(self) -> float:
        """Get bbox width."""
        return abs(self.x2 - self.x1)

    def height(self) -> float:
        """Get bbox height."""
        return abs(self.y2 - self.y1)

    def area(self) -> float:
        """Get bbox area."""
        return self.width() * self.height()


def split_text_bbox(
    text: str, bbox: Union[BBox, BBoxModel], add_jitter: bool = True
) -> list[tuple[str, Union[BBox, BBoxModel]]]:
    """
    Split text into words and distribute bounding boxes proportionally.
    Enhanced to work with both legacy BBox and new BBoxModel.

    Args:
        text: Input text to split
        bbox: Original bounding box for the entire text
        add_jitter: Whether to add small random jitter to word bboxes

    Returns:
        List of (word, bbox) tuples
    """
    words = text.strip().split()
    if not words:
        return []

    # Handle BBoxModel type
    if isinstance(bbox, BBoxModel):
        if len(words) == 1:
            return [(words[0], bbox)]  # BBoxModel doesn't have add_jitter yet

        # Calculate proportional widths based on character count
        total_chars = sum(len(word) for word in words)
        word_proportions = [len(word) / total_chars for word in words]

        # Distribute bboxes horizontally (assuming left-to-right layout)
        word_bboxes: list[tuple[str, Union[BBox, BBoxModel]]] = []
        current_x = bbox.x1
        bbox_width = bbox.x2 - bbox.x1

        for i, (word, proportion) in enumerate(zip(words, word_proportions)):
            word_width = bbox_width * proportion
            if i > 0:
                current_x += bbox_width * 0.02  # 2% gap
            word_bbox = BBoxModel(
                x1=current_x, y1=bbox.y1, x2=current_x + word_width, y2=bbox.y2
            )
            word_bboxes.append((word, word_bbox))
            current_x += word_width

        return word_bboxes

    # Handle legacy BBox type
    if len(words) == 1:
        return [(words[0], bbox.add_jitter() if add_jitter else bbox)]

    # Calculate proportional widths based on character count
    total_chars = sum(len(word) for word in words)
    word_proportions = [len(word) / total_chars for word in words]

    # Distribute bboxes horizontally (assuming left-to-right layout)
    word_bboxes_legacy: list[tuple[str, Union[BBox, BBoxModel]]] = []
    current_x = bbox.x1
    bbox_width = bbox.x2 - bbox.x1

    for i, (word, proportion) in enumerate(zip(words, word_proportions)):
        word_width = bbox_width * proportion
        if i > 0:
            current_x += bbox_width * 0.02  # 2% gap
        legacy_bbox = BBox(current_x, bbox.y1, current_x + word_width, bbox.y2)
        if add_jitter:
            legacy_bbox = legacy_bbox.add_jitter()
        word_bboxes_legacy.append((word, legacy_bbox))
        current_x += word_width

    return word_bboxes_legacy


def load_csv_data_as_models(csv_path: str) -> list[DataRecord]:
    """
    Load CSV data and convert to Pydantic DataRecord models.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of validated DataRecord models
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    records = []

    for _, row in df.iterrows():
        # Convert row to dictionary and handle NaN values
        row_dict = row.to_dict()
        for key, value in row_dict.items():
            if pd.isna(value):
                row_dict[key] = ""
            else:
                row_dict[key] = str(value)

        # Create DataRecord model with validation
        try:
            record = DataRecord(**row_dict)
            records.append(record)
        except Exception as e:
            print(f"Warning: Failed to validate row data: {e}")
            # Create with basic fields only
            record = DataRecord(
                hospital_name_text=row_dict.get("hospital_name_text", ""),
                doctor_name_text=row_dict.get("doctor_name_text", ""),
                additional_fields=row_dict,
            )
            records.append(record)

    return records


def split_text_into_words(
    text: str, bbox: BBox, add_jitter: bool = True
) -> list[tuple[str, BBox]]:
    """Split text into words with proportional bounding boxes."""
    words = text.split()

    if not words:
        return []

    if len(words) == 1:
        return [(words[0], bbox.add_jitter() if add_jitter else bbox)]

    # Calculate proportional widths based on character count
    total_chars = sum(len(word) for word in words)
    word_proportions = [len(word) / total_chars for word in words]

    # Distribute bboxes horizontally (assuming left-to-right layout)
    word_bboxes: list[tuple[str, BBox]] = []
    current_x = bbox.x1
    bbox_width = bbox.width()

    for i, (word, proportion) in enumerate(zip(words, word_proportions)):
        word_width = bbox_width * proportion

        # Add small gaps between words
        if i > 0:
            current_x += bbox_width * 0.02  # 2% gap

        word_bbox = BBox(current_x, bbox.y1, current_x + word_width, bbox.y2)

        if add_jitter:
            word_bbox = word_bbox.add_jitter()

        word_bboxes.append((word, word_bbox))
        current_x += word_width

    return word_bboxes


def load_layout_json(json_path: str) -> dict[str, list[float]]:
    """
    Load layout JSON file containing field bounding boxes.

    Args:
        json_path: Path to the layout JSON file

    Returns:
        Dictionary mapping field names to bbox coordinates [x1, y1, x2, y2]
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Layout JSON not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        layout_data = json.load(f)

    # Validate layout format
    for field_name, coords in layout_data.items():
        if not isinstance(coords, list) or len(coords) != 4:
            raise ValueError(f"Invalid bbox format for field '{field_name}': {coords}")

    result: dict[str, list[float]] = layout_data
    return result


def get_available_fonts(fonts_dir: str) -> list[str]:
    """
    Get list of available font files from the fonts directory.

    Args:
        fonts_dir: Path to directory containing font files

    Returns:
        List of font file paths
    """
    if not os.path.exists(fonts_dir):
        return []

    font_extensions = ["*.ttf", "*.otf", "*.ttc"]
    font_files = []

    for ext in font_extensions:
        font_files.extend(glob.glob(os.path.join(fonts_dir, ext)))
        font_files.extend(glob.glob(os.path.join(fonts_dir, "**", ext), recursive=True))

    return sorted(font_files)


def select_random_font(
    fonts_dir: str, default_font_size: int = 12
) -> Optional[ImageFont.FreeTypeFont]:
    """
    Select a random font from available fonts.

    Args:
        fonts_dir: Path to fonts directory
        default_font_size: Default font size to use

    Returns:
        PIL ImageFont object or None if no fonts available
    """
    font_files = get_available_fonts(fonts_dir)

    if not font_files:
        return None

    try:
        font_path = random.choice(font_files)
        return ImageFont.truetype(font_path, default_font_size)
    except OSError:
        return None


def ensure_dir_exists(dir_path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(dir_path, exist_ok=True)


def save_xfund_annotation(annotation_data: dict[str, Any], output_path: str) -> None:
    """
    Save annotation data in XFUND JSON format.

    Args:
        annotation_data: Dictionary containing XFUND annotation data
        output_path: Path where to save the JSON file
    """
    ensure_dir_exists(os.path.dirname(output_path))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotation_data, f, indent=2, ensure_ascii=False)


def load_csv_data(csv_path: str) -> list[dict[str, str]]:
    """
    Load CSV data as list of dictionaries.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of dictionaries, each representing a row
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    result: list[dict[str, str]] = df.to_dict("records")
    return result


def generate_unique_id(index: int, prefix: str = "") -> str:
    """
    Generate unique ID for dataset entries.

    Args:
        index: Numeric index
        prefix: Optional prefix for the ID

    Returns:
        Formatted ID string
    """
    return f"{prefix}{index:04d}" if prefix else f"{index:04d}"


def validate_image_size(
    img_path: str, min_width: int = 100, min_height: int = 100
) -> bool:
    """
    Validate that image meets minimum size requirements.

    Args:
        img_path: Path to image file
        min_width: Minimum required width
        min_height: Minimum required height

    Returns:
        True if image is valid, False otherwise
    """
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            return width >= min_width and height >= min_height
    except Exception:
        return False


def apply_bbox_transform(
    bbox: Union[BBox, BBoxModel],
    transform_matrix: np.ndarray,
    img_shape: tuple[int, int],
) -> Union[BBox, BBoxModel]:
    """
    Apply transformation matrix to bounding box coordinates.
    Enhanced to work with both legacy BBox and BBoxModel.

    Args:
        bbox: Original bounding box
        transform_matrix: 3x3 transformation matrix
        img_shape: (height, width) of the image

    Returns:
        Transformed bounding box of the same type
    """
    # Convert bbox corners to homogeneous coordinates
    corners = np.array(
        [
            [bbox.x1, bbox.y1, 1],
            [bbox.x2, bbox.y1, 1],
            [bbox.x2, bbox.y2, 1],
            [bbox.x1, bbox.y2, 1],
        ]
    ).T

    # Apply transformation
    transformed_corners = transform_matrix @ corners

    # Get bounding box of transformed corners
    x_coords = transformed_corners[0, :]
    y_coords = transformed_corners[1, :]

    new_x1 = max(0, min(x_coords))
    new_y1 = max(0, min(y_coords))
    new_x2 = min(img_shape[1], max(x_coords))
    new_y2 = min(img_shape[0], max(y_coords))

    # Return same type as input
    if isinstance(bbox, BBoxModel):
        return BBoxModel(x1=new_x1, y1=new_y1, x2=new_x2, y2=new_y2)
    else:
        return BBox(new_x1, new_y1, new_x2, new_y2)


# Enhanced Pydantic utility functions


def save_xfund_annotation_pydantic(
    annotation: XFUNDAnnotation, output_path: str
) -> None:
    """
    Save XFUND annotation using Pydantic model.

    Args:
        annotation: Validated XFUNDAnnotation model
        output_path: Path to save the JSON annotation
    """
    annotation_dict = {"form": [entity.model_dump() for entity in annotation.form]}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotation_dict, f, indent=2, ensure_ascii=False)


def create_xfund_entity_from_text(
    entity_id: int,
    text: str,
    bbox: Union[BBox, BBoxModel],
    label: str = "OTHER",
    split_words: bool = True,
) -> XFUNDEntity:
    """
    Create XFUND entity from text and bbox with validation.

    Args:
        entity_id: Unique entity ID
        text: Text content
        bbox: Bounding box coordinates
        label: Entity label (HEADER, QUESTION, ANSWER, OTHER)
        split_words: Whether to split text into individual words

    Returns:
        Validated XFUNDEntity
    """
    # Convert BBox to BBoxModel if needed
    bbox_model = bbox.to_pydantic() if isinstance(bbox, BBox) else bbox

    # Split into words if requested
    words = text.strip().split() if split_words else None

    return XFUNDEntity(
        id=entity_id, text=text, bbox=bbox_model, label=label, words=words
    )


def validate_annotation_quality(
    annotation: XFUNDAnnotation, min_text_length: int = 2, min_bbox_area: float = 100.0
) -> list[str]:
    """
    Validate annotation quality and return list of issues found.

    Args:
        annotation: XFUNDAnnotation to validate
        min_text_length: Minimum required text length
        min_bbox_area: Minimum required bbox area

    Returns:
        List of validation error messages
    """
    issues = []

    if not annotation.form:
        issues.append("Annotation contains no entities")
        return issues

    for entity in annotation.form:
        # Check text length
        if len(entity.text.strip()) < min_text_length:
            issues.append(f"Entity {entity.id} has text too short: '{entity.text}'")

        # Check bbox area
        if entity.bbox.area() < min_bbox_area:
            issues.append(
                f"Entity {entity.id} has bbox area too small: {entity.bbox.area()}"
            )

        # Check bbox validity
        if entity.bbox.x1 >= entity.bbox.x2 or entity.bbox.y1 >= entity.bbox.y2:
            issues.append(f"Entity {entity.id} has invalid bbox coordinates")

    return issues


def merge_annotations(annotations: list[XFUNDAnnotation]) -> XFUNDAnnotation:
    """
    Merge multiple XFUND annotations into one.

    Args:
        annotations: List of XFUNDAnnotation objects to merge

    Returns:
        Merged XFUNDAnnotation
    """
    if not annotations:
        raise ValueError("Cannot merge empty list of annotations")

    merged_entities = []
    entity_id = 0

    for annotation in annotations:
        for entity in annotation.form:
            # Create new entity with sequential ID
            merged_entity = XFUNDEntity(
                id=entity_id,
                text=entity.text,
                bbox=entity.bbox,
                label=entity.label,
                words=entity.words,
                linking=entity.linking,
            )
            merged_entities.append(merged_entity)
            entity_id += 1

    # Use image path from first annotation
    return XFUNDAnnotation(form=merged_entities, image_path=annotations[0].image_path)


def load_config_with_validation(config_path: str) -> GeneratorConfig:
    """
    Load and validate configuration file.

    Args:
        config_path: Path to configuration JSON file

    Returns:
        Validated GeneratorConfig

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If config file doesn't exist
    """
    from .models import validate_config_file

    validation_result = validate_config_file(config_path)

    if not validation_result.is_valid:
        error_msg = "Configuration validation failed:\n"
        for error in validation_result.errors:
            error_msg += f"  - {error}\n"
        raise ValueError(error_msg)

    if validation_result.warnings:
        print("Configuration warnings:")
        for warning in validation_result.warnings:
            print(f"  - {warning}")

    return GeneratorConfig.from_json_file(config_path)


def convert_legacy_bbox_data(
    legacy_data: dict[str, list[float]],
) -> dict[str, BBoxModel]:
    """
    Convert legacy bbox data to Pydantic models.

    Args:
        legacy_data: Dictionary with field names mapping to [x1, y1, x2, y2] lists

    Returns:
        Dictionary with field names mapping to BBoxModel objects
    """
    converted = {}

    for field_name, coords in legacy_data.items():
        if len(coords) != 4:
            raise ValueError(f"Invalid bbox coordinates for {field_name}: {coords}")

        converted[field_name] = BBoxModel(
            x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]
        )

    return converted


def calculate_iou(bbox1: BBox, bbox2: BBox) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        IoU score between 0 and 1
    """
    # Calculate intersection area
    x1 = max(bbox1.x1, bbox2.x1)
    y1 = max(bbox1.y1, bbox2.y1)
    x2 = min(bbox1.x2, bbox2.x2)
    y2 = min(bbox1.y2, bbox2.y2)

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Calculate union area
    area1 = bbox1.area()
    area2 = bbox2.area()
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


# Field mapping for common medical document fields
FIELD_MAPPINGS = {
    "hospital_name": [
        "hospital name",
        "hospital_name",
        "clinic name",
        "medical center",
    ],
    "hospital_address": ["hospital address", "hospital_address", "clinic address"],
    "doctor_name": ["doctor name", "doctor_name", "physician name", "dr name"],
    "patient_name": ["patient name", "patient_name", "patient"],
    "diagnose": ["diagnose", "diagnosis", "condition"],
    "doctor_comment": [
        "doctor comment",
        "doctor_comment",
        "prescription",
        "treatment",
        "medication",
    ],
}


def normalize_field_name(field_name: str) -> str:
    """
    Normalize field name to standard format.

    Args:
        field_name: Original field name

    Returns:
        Normalized field name
    """
    field_name = field_name.lower().strip()

    for standard_name, variants in FIELD_MAPPINGS.items():
        if field_name in variants:
            return standard_name

    return field_name.replace(" ", "_")
