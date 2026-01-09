"""
XFUND Form Integration Module
Utilizes the form classes to generate image annotations in XFUND format.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .docx_utils import process_docx_template
from .form import Word, XFUNDAnnotation, XFUNDDataset
from .models import DataRecord, GeneratorConfig
from .utils import ensure_dir_exists

logger = logging.getLogger(__name__)


class XFUNDFormGenerator:
    """
    Generator that creates XFUND format annotations using the form classes.
    Bridges the gap between template processing and standardized XFUND output.
    """

    def __init__(self, config: GeneratorConfig):
        """Initialize the XFUND form generator."""
        self.config = config
        self.annotation_id_counter = 1

    def generate_xfund_from_template(
        self, template_path: str, data_records: list[DataRecord], output_dir: str
    ) -> tuple[str, XFUNDDataset]:
        """
        Generate XFUND format annotation from template and data.

        Args:
            template_path: Path to DOCX template
            data_records: List of data records for the template
            output_dir: Output directory for generated files

        Returns:
            Tuple of (image_path, xfund_dataset)
        """
        template_name = Path(template_path).stem

        # Group data by template
        template_data: dict[str, str] = {}
        for record in data_records:
            if (
                record.template_name == template_name
                and record.field_name is not None
                and record.field_value is not None
            ):
                template_data[record.field_name] = record.field_value

        if not template_data:
            raise ValueError(f"No data found for template: {template_name}")

        # Generate image from template
        output_image_path = os.path.join(
            output_dir, f"{template_name}_{self.annotation_id_counter:04d}.png"
        )

        logger.info(f"Processing template: {template_name}")
        image_path, image_size = process_docx_template(
            template_path=template_path,
            data=template_data,
            output_image_path=output_image_path,
            dpi=self.config.image_dpi,
        )

        # Create XFUND annotations
        xfund_annotations = self._create_xfund_annotations(
            data_records, template_name, image_size
        )

        # Create XFUND dataset
        xfund_dataset = XFUNDDataset(annotations=xfund_annotations)

        return image_path, xfund_dataset

    def _create_xfund_annotations(
        self,
        data_records: list[DataRecord],
        template_name: str,
        image_size: tuple[int, int],
    ) -> list[XFUNDAnnotation]:
        """
        Create XFUND annotations from data records.

        Args:
            data_records: List of data records
            template_name: Name of the template
            image_size: Size of the generated image (width, height)

        Returns:
            List of XFUND annotations
        """
        annotations = []

        for record in data_records:
            if record.template_name != template_name:
                continue
            if record.field_name is None or record.field_value is None:
                continue

            # Parse bounding box coordinates
            bbox_coords = record.get_bbox_coordinates()

            # Validate bbox is within image bounds
            bbox_coords = self._validate_bbox_bounds(bbox_coords, image_size)

            # Determine label type based on field name
            label = self._determine_label_type(record.field_name)

            # Create word-level annotations
            words = self._create_word_annotations(record.field_value, bbox_coords)

            # Create XFUND annotation
            annotation = XFUNDAnnotation(
                id=self.annotation_id_counter,
                box=list(bbox_coords),
                text=record.field_value,
                label=label,
                words=words,
                linking=[],  # Will be populated by relationship detection
            )

            annotations.append(annotation)
            self.annotation_id_counter += 1

        # Add question-answer linking
        self._add_question_answer_linking(annotations)

        return annotations

    def _validate_bbox_bounds(
        self, bbox_coords: tuple[int, int, int, int], image_size: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        """Validate and clamp bounding box to image bounds."""
        x1, y1, x2, y2 = bbox_coords
        width, height = image_size

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        return (x1, y1, x2, y2)

    def _determine_label_type(self, field_name: str) -> str:
        """
        Determine XFUND label type based on field name.

        Args:
            field_name: Field identifier

        Returns:
            XFUND label type
        """
        field_name_lower = field_name.lower()

        # Question indicators
        question_keywords = ["question", "label", "prompt", "field_name", "key"]
        if any(keyword in field_name_lower for keyword in question_keywords):
            return "question"

        # Answer indicators
        answer_keywords = ["answer", "value", "response", "data", "content"]
        if any(keyword in field_name_lower for keyword in answer_keywords):
            return "answer"

        # Default based on common patterns
        if field_name_lower.endswith("_label") or field_name_lower.endswith("_name"):
            return "question"
        elif field_name_lower.endswith("_value") or field_name_lower.endswith("_data"):
            return "answer"

        # Default to "other" for ambiguous fields
        return "other"

    def _create_word_annotations(
        self, text: str, bbox_coords: tuple[int, int, int, int]
    ) -> list[Word]:
        """
        Create word-level annotations from text and bounding box.

        Args:
            text: Text content
            bbox_coords: Overall bounding box coordinates

        Returns:
            List of word annotations
        """
        words = text.split()
        if not words:
            return []

        x1, y1, x2, y2 = bbox_coords
        bbox_width = x2 - x1

        word_annotations = []

        if len(words) == 1:
            # Single word takes the entire bbox
            word_annotations.append(Word(box=[x1, y1, x2, y2], text=words[0]))
        else:
            # Distribute words horizontally
            total_chars = sum(len(word) for word in words)
            current_x = x1

            for word in words:
                word_width = int(bbox_width * (len(word) / total_chars))
                word_x2 = min(current_x + word_width, x2)

                word_annotations.append(
                    Word(box=[current_x, y1, word_x2, y2], text=word)
                )

                current_x = word_x2

        return word_annotations

    def _add_question_answer_linking(self, annotations: list[XFUNDAnnotation]) -> None:
        """
        Add question-answer linking based on spatial and semantic relationships.

        Args:
            annotations: List of annotations to process for linking
        """
        questions = [ann for ann in annotations if ann.label == "question"]
        answers = [ann for ann in annotations if ann.label == "answer"]

        for question in questions:
            # Find spatially close answers
            nearby_answers = self._find_nearby_answers(question, answers)

            # Create linking
            for answer in nearby_answers:
                if question.id is not None and answer.id is not None:
                    question.linking.append([question.id, answer.id])

    def _find_nearby_answers(
        self,
        question: XFUNDAnnotation,
        answers: list[XFUNDAnnotation],
        max_distance: int = 100,
    ) -> list[XFUNDAnnotation]:
        """
        Find answer annotations that are spatially close to a question.

        Args:
            question: Question annotation
            answers: List of answer annotations
            max_distance: Maximum pixel distance to consider "nearby"

        Returns:
            List of nearby answer annotations
        """
        if not question.box:
            return []

        q_x1, q_y1, q_x2, q_y2 = question.box
        q_center_x = (q_x1 + q_x2) / 2
        q_center_y = (q_y1 + q_y2) / 2

        nearby_answers = []

        for answer in answers:
            if not answer.box:
                continue

            a_x1, a_y1, a_x2, a_y2 = answer.box
            a_center_x = (a_x1 + a_x2) / 2
            a_center_y = (a_y1 + a_y2) / 2

            # Calculate distance
            distance = np.sqrt(
                (q_center_x - a_center_x) ** 2 + (q_center_y - a_center_y) ** 2
            )

            if distance <= max_distance:
                nearby_answers.append(answer)

        # Sort by distance and return closest ones
        def get_distance(a: XFUNDAnnotation) -> float:
            if a.box is None:
                return float("inf")
            return float(
                np.sqrt(
                    ((q_x1 + q_x2) / 2 - (a.box[0] + a.box[2]) / 2) ** 2
                    + ((q_y1 + q_y2) / 2 - (a.box[1] + a.box[3]) / 2) ** 2
                )
            )

        nearby_answers.sort(key=get_distance)

        return nearby_answers[:3]  # Return up to 3 closest answers

    def save_xfund_annotation(
        self, xfund_dataset: XFUNDDataset, output_path: str, image_path: str
    ) -> None:
        """
        Save XFUND dataset annotation to file.

        Args:
            xfund_dataset: XFUND dataset to save
            output_path: Output annotation file path
            image_path: Corresponding image file path
        """
        ensure_dir_exists(os.path.dirname(output_path))

        # Create XFUND format with image info
        xfund_data = json.loads(xfund_dataset.to_json())
        xfund_data["image"] = {
            "path": os.path.basename(image_path),
            "width": 0,  # Will be updated with actual image size
            "height": 0,
        }

        # Get actual image size
        try:
            with Image.open(image_path) as img:
                xfund_data["image"]["width"] = img.width
                xfund_data["image"]["height"] = img.height
        except Exception as e:
            logger.warning(f"Could not get image size: {e}")

        # Save annotation
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(xfund_data, f, indent=2, ensure_ascii=False)

        logger.info(f"XFUND annotation saved to: {output_path}")

    def generate_batch_xfund_annotations(
        self, templates_dir: str, data_records: list[DataRecord], output_dir: str
    ) -> list[dict[str, Any]]:
        """
        Generate XFUND annotations for multiple templates in batch.

        Args:
            templates_dir: Directory containing DOCX templates
            data_records: List of all data records
            output_dir: Output directory for generated files

        Returns:
            List of generation results
        """
        # Create output directories
        images_dir = os.path.join(output_dir, "images")
        annotations_dir = os.path.join(output_dir, "annotations")
        ensure_dir_exists(images_dir)
        ensure_dir_exists(annotations_dir)

        # Group records by template
        template_groups: dict[str, list[DataRecord]] = {}
        for record in data_records:
            if record.template_name is None:
                continue
            if record.template_name not in template_groups:
                template_groups[record.template_name] = []
            template_groups[record.template_name].append(record)

        results = []

        for template_name, records in template_groups.items():
            template_path = os.path.join(templates_dir, f"{template_name}.docx")

            if not os.path.exists(template_path):
                logger.warning(f"Template not found: {template_path}")
                continue

            try:
                # Generate XFUND annotation
                image_path, xfund_dataset = self.generate_xfund_from_template(
                    template_path, records, images_dir
                )

                # Save annotation
                annotation_filename = f"{Path(image_path).stem}.json"
                annotation_path = os.path.join(annotations_dir, annotation_filename)

                self.save_xfund_annotation(xfund_dataset, annotation_path, image_path)

                # Record result
                result = {
                    "template_name": template_name,
                    "image_path": image_path,
                    "annotation_path": annotation_path,
                    "annotations_count": len(xfund_dataset.annotations),
                    "qa_pairs": len(xfund_dataset.get_flat_qa_pairs()),
                    "status": "success",
                }

                results.append(result)
                logger.info(f"Successfully processed template: {template_name}")

            except Exception as e:
                logger.error(f"Error processing template {template_name}: {e}")
                results.append(
                    {
                        "template_name": template_name,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        return results


def demo_xfund_form_generation():
    """Demonstrate XFUND form generation."""
    print("🎯 XFUND Form Generation Demo\n")

    # Sample configuration
    from .models import get_default_config

    config = get_default_config()
    config.output_dir = "output/xfund_forms"

    # Sample data records
    sample_records = [
        DataRecord(
            template_name="medical_form",
            field_name="patient_name_label",
            field_value="Patient Name:",
            bbox="50,50,150,70",
        ),
        DataRecord(
            template_name="medical_form",
            field_name="patient_name_value",
            field_value="John Doe",
            bbox="160,50,250,70",
        ),
        DataRecord(
            template_name="medical_form",
            field_name="age_label",
            field_value="Age:",
            bbox="50,80,100,100",
        ),
        DataRecord(
            template_name="medical_form",
            field_name="age_value",
            field_value="32",
            bbox="110,80,140,100",
        ),
    ]

    # Create generator
    XFUNDFormGenerator(config)

    print(f"📋 Created {len(sample_records)} sample records")
    print("🏭 Generating XFUND annotations...")

    # This would normally process real templates
    print("💡 Note: This demo shows the structure. Real usage requires:")
    print("   - DOCX template files")
    print("   - LibreOffice for conversion")
    print("   - Proper template layout JSON files")

    # Show what the output would look like
    print("\n📊 Expected XFUND Output Structure:")

    # Create sample XFUND annotation manually for demonstration
    from form import Word, XFUNDAnnotation, XFUNDDataset

    sample_annotations = [
        XFUNDAnnotation(
            id=1,
            box=[50, 50, 150, 70],
            text="Patient Name:",
            label="question",
            words=[
                Word(box=[50, 50, 95, 70], text="Patient"),
                Word(box=[100, 50, 150, 70], text="Name:"),
            ],
            linking=[[1, 2]],
        ),
        XFUNDAnnotation(
            id=2,
            box=[160, 50, 250, 70],
            text="John Doe",
            label="answer",
            words=[
                Word(box=[160, 50, 195, 70], text="John"),
                Word(box=[200, 50, 250, 70], text="Doe"),
            ],
        ),
    ]

    sample_dataset = XFUNDDataset(annotations=sample_annotations)
    print("Sample XFUND JSON:")
    print(sample_dataset.to_json(indent=2)[:500] + "...")

    print("\n✅ XFUND form generation demo completed!")


if __name__ == "__main__":
    demo_xfund_form_generation()
