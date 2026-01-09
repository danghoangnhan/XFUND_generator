"""
Word-level rendering and bounding box generation for XFUND dataset.
Generates word-level annotations from layout templates and field data.
"""

import logging
import os
from typing import Any, Optional, Union

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

from .models import (
    AnnotationValidationResult,
    WordAnnotation,
    validate_annotations as pydantic_validate_annotations,
)
from .utils import (
    BBox,
    load_layout_json,
    normalize_field_name,
    select_random_font,
    split_text_bbox,
)

logger = logging.getLogger(__name__)


class WordRenderer:
    """Generates word-level bounding boxes and annotations."""

    def __init__(
        self,
        layout_json_path: str,
        fonts_dir: Optional[str] = None,
        target_size: int = 1000,
    ):
        """
        Initialize word renderer with layout configuration.

        Args:
            layout_json_path: Path to layout JSON file
            fonts_dir: Optional directory containing fonts
            target_size: Target size for XFUND normalization (default 1000)
        """
        self.layout_data = load_layout_json(layout_json_path)
        self.fonts_dir = fonts_dir
        self.target_size = target_size

        logger.info(f"Loaded layout with {len(self.layout_data)} fields")

    def generate_word_annotations(
        self,
        field_data: dict[str, str],
        image_size: tuple[int, int],
        add_jitter: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Generate word-level XFUND annotations from field data.

        Uses WordAnnotation Pydantic model internally for validation.

        Args:
            field_data: Dictionary of field names to text values
            image_size: Tuple of (width, height) of the image
            add_jitter: Whether to add small random jitter to bboxes

        Returns:
            List of XFUND annotation dictionaries
        """
        annotations: list[WordAnnotation] = []
        img_width, img_height = image_size

        for field_name, text_value in field_data.items():
            if not text_value or not text_value.strip():
                continue

            # Get field bbox from layout
            field_bbox = self._get_field_bbox(field_name)
            if field_bbox is None:
                logger.warning(f"No layout found for field: {field_name}")
                continue

            # Split text into words with individual bboxes
            word_bboxes = split_text_bbox(str(text_value), field_bbox, add_jitter)

            # Convert to XFUND format using WordAnnotation model
            for word_text, word_bbox in word_bboxes:
                # Normalize bbox to XFUND scale
                normalized_bbox = word_bbox.normalize(
                    img_width, img_height, self.target_size
                )

                # Create validated WordAnnotation
                annotation = WordAnnotation(
                    text=word_text,
                    bbox=normalized_bbox.to_xfund_format(),
                    label=normalize_field_name(field_name),
                )
                annotations.append(annotation)

        logger.info(f"Generated {len(annotations)} word annotations")
        # Convert to dicts for backward compatibility
        return [ann.to_dict() for ann in annotations]

    def _get_field_bbox(self, field_name: str) -> Optional[BBox]:
        """
        Get bounding box for a field from layout data.

        Args:
            field_name: Name of the field

        Returns:
            BBox object or None if field not found
        """
        # Try exact match first
        if field_name in self.layout_data:
            coords = self.layout_data[field_name]
            return BBox(coords[0], coords[1], coords[2], coords[3])

        # Try normalized field name
        normalized_name = normalize_field_name(field_name)
        if normalized_name in self.layout_data:
            coords = self.layout_data[normalized_name]
            return BBox(coords[0], coords[1], coords[2], coords[3])

        # Try field mapping variants
        for layout_field in self.layout_data:
            normalized_layout_field = normalize_field_name(layout_field)
            if normalized_layout_field == normalized_name:
                coords = self.layout_data[layout_field]
                return BBox(coords[0], coords[1], coords[2], coords[3])

        return None

    def render_debug_overlay(
        self,
        image_path: str,
        annotations: list[dict[str, Any]],
        output_path: str,
        show_labels: bool = True,
    ) -> str:
        """
        Render debug overlay showing bounding boxes on image.

        Args:
            image_path: Path to the source image
            annotations: List of XFUND annotations
            output_path: Path for output debug image
            show_labels: Whether to show field labels on bboxes

        Returns:
            Path to the debug overlay image
        """
        try:
            # Load image
            with Image.open(image_path) as img_file:
                img = img_file.convert("RGB")
                draw = ImageDraw.Draw(img)
                img_width, img_height = img.size

                # Color map for different labels
                colors = [
                    (255, 0, 0),  # Red
                    (0, 255, 0),  # Green
                    (0, 0, 255),  # Blue
                    (255, 255, 0),  # Yellow
                    (255, 0, 255),  # Magenta
                    (0, 255, 255),  # Cyan
                ]

                label_colors = {}
                color_idx = 0

                # Load font for labels
                font: Union[FreeTypeFont, ImageFont.ImageFont]
                try:
                    selected_font = (
                        select_random_font(self.fonts_dir, 16)
                        if self.fonts_dir
                        else None
                    )
                    if selected_font is not None:
                        font = selected_font
                    else:
                        font = ImageFont.load_default()
                except OSError:
                    font = ImageFont.load_default()

                # Draw bounding boxes
                for annotation in annotations:
                    label = annotation["label"]
                    bbox_coords = annotation["bbox"]
                    text = annotation["text"]

                    # Assign color to label
                    if label not in label_colors:
                        label_colors[label] = colors[color_idx % len(colors)]
                        color_idx += 1

                    color = label_colors[label]

                    # Denormalize bbox coordinates
                    bbox = BBox(
                        bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]
                    )
                    actual_bbox = bbox.denormalize(
                        img_width, img_height, self.target_size
                    )

                    # Draw bounding box
                    draw.rectangle(
                        [
                            actual_bbox.x1,
                            actual_bbox.y1,
                            actual_bbox.x2,
                            actual_bbox.y2,
                        ],
                        outline=color,
                        width=2,
                    )

                    # Draw text and label
                    if show_labels:
                        label_text = f"{text} ({label})"
                        draw.text(
                            (actual_bbox.x1, actual_bbox.y1 - 20),
                            label_text,
                            fill=color,
                            font=font,
                        )

                # Save debug image
                img.save(output_path)
                logger.info(f"Debug overlay saved: {output_path}")
                return output_path

        except Exception as e:
            logger.error(f"Error creating debug overlay: {e}")
            raise

    def validate_annotations(
        self,
        annotations: list[dict[str, Any]],
        image_size: tuple[int, int],  # noqa: ARG002
    ) -> AnnotationValidationResult:
        """
        Validate generated annotations for quality and consistency.

        Uses Pydantic models for validation with proper type safety.

        Args:
            annotations: List of XFUND annotations
            image_size: Image dimensions (unused, kept for API compatibility)

        Returns:
            AnnotationValidationResult with validation status and statistics
        """
        return pydantic_validate_annotations(
            annotations,
            target_size=self.target_size,
            check_overlaps=True,
        )

    def create_xfund_entry(
        self, entry_id: str, image_filename: str, annotations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Create complete XFUND format entry.

        Args:
            entry_id: Unique ID for the entry
            image_filename: Name of the image file (relative to images directory)
            annotations: List of word-level annotations

        Returns:
            Complete XFUND format dictionary
        """
        return {
            "id": entry_id,
            "image": f"images/{image_filename}",
            "annotations": annotations,
        }

    def get_layout_fields(self) -> list[str]:
        """Get list of all fields defined in the layout."""
        return list(self.layout_data.keys())

    def estimate_text_bbox(
        self, text: str, font_size: int = 12, font_path: Optional[str] = None
    ) -> BBox:
        """
        Estimate bounding box size for text rendering.

        Args:
            text: Text to estimate bbox for
            font_size: Font size for estimation
            font_path: Optional path to specific font file

        Returns:
            Estimated bounding box
        """
        try:
            font: Union[FreeTypeFont, ImageFont.ImageFont]
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()

            # Create temporary image to measure text
            temp_img = Image.new("RGB", (1000, 1000), "white")
            draw = ImageDraw.Draw(temp_img)

            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            return BBox(0, 0, width, height)

        except Exception as e:
            logger.warning(f"Error estimating text bbox: {e}")
            # Fallback estimation
            char_width = font_size * 0.6
            char_height = font_size * 1.2
            width = len(text) * char_width
            height = char_height
            return BBox(0, 0, width, height)


# Alias for backwards compatibility
DocumentRenderer = WordRenderer


def generate_layout_from_template(
    template_analysis: dict[str, Any],
    image_size: tuple[int, int],
    field_mapping: Optional[dict[str, str]] = None,
) -> dict[str, list[float]]:
    """
    Generate layout JSON from template analysis.
    This is a helper function for creating initial layout files.

    Args:
        template_analysis: Analysis results from docx template validation
        image_size: Size of the rendered image
        field_mapping: Optional mapping from placeholders to field names

    Returns:
        Layout dictionary suitable for saving as JSON
    """
    layout = {}
    placeholders = template_analysis.get("placeholders", [])

    img_width, img_height = image_size

    # Simple grid layout estimation
    # This is a basic implementation - real layouts should be manually defined
    for i, placeholder in enumerate(placeholders):
        # Map placeholder to standard field name if mapping provided
        field_name = (
            field_mapping.get(placeholder, placeholder)
            if field_mapping
            else placeholder
        )
        field_name = normalize_field_name(field_name)

        # Simple grid positioning (2 columns)
        col = i % 2
        row = i // 2

        # Calculate position
        x_margin = img_width * 0.1
        y_margin = img_height * 0.1
        col_width = (img_width - 2 * x_margin) / 2
        row_height = 40  # Fixed height per row

        x1 = x_margin + col * col_width
        y1 = y_margin + row * row_height
        x2 = x1 + col_width * 0.8  # Leave some gap
        y2 = y1 + row_height * 0.8

        layout[field_name] = [x1, y1, x2, y2]

    return layout


def create_sample_layout() -> dict[str, list[float]]:
    """
    Create a sample layout configuration for medical documents.

    Returns:
        Sample layout dictionary
    """
    return {
        "hospital_name": [100, 50, 400, 100],
        "hospital_address": [100, 120, 500, 170],
        "doctor_name": [100, 200, 350, 250],
        "patient_name": [400, 200, 650, 250],
        "diagnose": [100, 300, 700, 400],
        "doctor_comment": [100, 450, 700, 550],
    }
