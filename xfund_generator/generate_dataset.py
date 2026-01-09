"""
Main dataset generation script for XFUND-style OCR dataset.
Orchestrates the complete pipeline from DOCX templates to XFUND annotations.
Enhanced with Pydantic support for better type safety and validation.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2

from .augmentations import (
    DocumentAugmenter,
    validate_augmentation_quality,
)
from .docx_utils import (
    check_libreoffice_installed,
    process_docx_template,
    validate_docx_template,
)
from .models import (
    DataRecord,
    GenerationResult,
    GeneratorConfig,
    XFUNDAnnotation,
)
from .renderer import WordRenderer, create_sample_layout
from .utils import (
    ensure_dir_exists,
    generate_unique_id,
    load_csv_data,
    load_csv_data_as_models,
    save_xfund_annotation,
    save_xfund_annotation_pydantic,
    validate_image_size,
)
from .xfund_form_integration import XFUNDFormGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class XFUNDGenerator:
    """Main class for XFUND dataset generation with Pydantic support."""

    def __init__(self, config: GeneratorConfig):
        """
        Initialize XFUND generator with validated configuration.

        Args:
            config: Validated GeneratorConfig object
        """
        self.config = config
        self.templates_dir = config.templates_dir
        self.csv_path = config.csv_path
        self.output_dir = config.output_dir
        self.fonts_dir = config.fonts_dir

        # Ensure output directories exist
        self.images_dir = os.path.join(self.output_dir, "images")
        self.annotations_dir = os.path.join(self.output_dir, "annotations")
        ensure_dir_exists(self.images_dir)
        ensure_dir_exists(self.annotations_dir)

        # Initialize components
        self.augmenter = None
        if config.enable_augmentations:
            aug_config = config.get_augmentation_config()
            # Convert to dict for compatibility with existing augmenter
            aug_config_dict = aug_config.model_dump()
            self.augmenter = DocumentAugmenter(**aug_config_dict)

        # Initialize form generator for XFUND format
        self.form_generator = XFUNDFormGenerator(config)

        logger.info("Initialized XFUND generator with Pydantic validation")
        logger.info(f"Templates: {self.templates_dir}")
        logger.info(f"CSV data: {self.csv_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Document type: {config.document_type}")

    def generate_dataset(self) -> GenerationResult:
        """
        Generate complete XFUND dataset with full validation.

        Returns:
            Validated GenerationResult object with statistics and results
        """
        start_time = time.time()
        logger.info("Starting XFUND dataset generation with Pydantic validation...")

        # Initialize result object
        result = GenerationResult(success=True, total_records=0, templates_used=0)

        # Load CSV data as validated models
        try:
            csv_records = load_csv_data_as_models(self.csv_path)
            result.total_records = len(csv_records)
            logger.info(f"Loaded and validated {len(csv_records)} records from CSV")
        except Exception as e:
            logger.error(f"Failed to load and validate CSV data: {e}")
            result.success = False
            result.add_error(str(e))
            return result

        # Find available templates
        templates = self._find_templates()
        if not templates:
            logger.error("No valid templates found")
            result.success = False
            result.add_error("No templates found")
            return result

        result.templates_used = len(templates)
        logger.info(f"Found {len(templates)} validated template(s)")

        # Generate dataset entries
        for i, data_record in enumerate(csv_records):
            try:
                # Select template (round-robin or random)
                template_info = templates[i % len(templates)]

                # Generate entry with validation
                entry_result = self._generate_single_entry_validated(
                    data_record, template_info, i
                )

                if entry_result["success"]:
                    result.generated_entries += 1
                    logger.info(
                        f"Generated validated entry {i + 1}/{len(csv_records)}: {entry_result['entry_id']}"
                    )
                else:
                    result.add_error(
                        f"Entry {i}: {entry_result.get('error', 'Unknown error')}"
                    )
                    logger.warning(
                        f"Failed to generate entry {i + 1}: {entry_result.get('error')}"
                    )

            except Exception as e:
                result.add_error(f"Entry {i}: {str(e)}")
                logger.error(f"Error generating entry {i + 1}: {e}")

        # Calculate final statistics
        end_time = time.time()
        result.generation_time = end_time - start_time
        result.output_paths = {
            "images": self.images_dir,
            "annotations": self.annotations_dir,
        }

        # Log summary
        logger.info("Dataset generation completed!")
        logger.info(f"Generated: {result.generated_entries}")
        logger.info(f"Failed: {result.failed_entries}")
        logger.info(f"Success rate: {result.success_rate:.1f}%")
        logger.info(f"Total time: {result.generation_time:.2f}s")

        if result.errors:
            logger.warning(f"Errors encountered: {len(result.errors)}")
            for error in result.errors[:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")

        return result

    def _find_templates(self) -> list[dict[str, str]]:
        """
        Find and validate available templates.

        Returns:
            List of template information dictionaries
        """
        templates: list[dict[str, str]] = []

        if not os.path.exists(self.templates_dir):
            logger.error(f"Templates directory not found: {self.templates_dir}")
            return templates

        # Find DOCX files
        docx_files = list(Path(self.templates_dir).glob("*.docx"))

        for docx_path in docx_files:
            template_name = docx_path.stem
            layout_path = docx_path.parent / f"{template_name}_layout.json"

            # Validate template
            template_validation = validate_docx_template(str(docx_path))
            if not template_validation.valid:
                logger.warning(
                    f"Invalid template {docx_path}: {template_validation.error}"
                )
                continue

            # Check for layout file
            if not layout_path.exists():
                logger.warning(f"No layout file found for {template_name}")
                logger.info(f"Expected layout file: {layout_path}")
                logger.info(
                    f"Creating basic layout for {template_name}. Please customize it!"
                )

                # Create basic sample layout
                sample_layout = create_sample_layout()
                with open(layout_path, "w") as f:
                    json.dump(sample_layout, f, indent=2)
                logger.info(
                    f"Created {template_name}_layout.json - please review and adjust coordinates"
                )

            # Validate layout file format
            try:
                with open(layout_path) as f:
                    layout_data = json.load(f)

                # Check if all required fields are present
                required_fields = [
                    "hospital_name",
                    "hospital_address",
                    "doctor_name",
                    "patient_name",
                    "diagnose",
                    "doctor_command",
                ]
                missing_fields = [
                    field for field in required_fields if field not in layout_data
                ]

                if missing_fields:
                    logger.warning(
                        f"Layout {layout_path} missing fields: {missing_fields}"
                    )
                else:
                    logger.info(f"✓ Layout validated for {template_name}")

            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Invalid layout file {layout_path}: {e}")
                continue

            templates.append(
                {
                    "name": template_name,
                    "docx_path": str(docx_path),
                    "layout_path": str(layout_path),
                }
            )

            logger.info(f"Found template: {template_name}")

        return templates

    def _generate_single_entry(
        self, data_row: dict[str, str], template_info: dict[str, str], index: int
    ) -> dict[str, Any]:
        """
        Generate a single dataset entry.

        Args:
            data_row: CSV data for this entry
            template_info: Template information
            index: Entry index

        Returns:
            Generation result dictionary
        """
        entry_id = generate_unique_id(index)

        try:
            # Step 1: Fill DOCX template and convert to PNG
            image_filename = f"{entry_id}.png"
            image_path = os.path.join(self.images_dir, image_filename)

            image_path, image_size = process_docx_template(
                template_info["docx_path"],
                data_row,
                image_path,
                dpi=self.config.image_dpi,
            )

            # Validate image
            if not validate_image_size(image_path):
                return {
                    "success": False,
                    "error": "Generated image too small or invalid",
                }

            # Step 2: Generate word-level annotations
            renderer = WordRenderer(
                template_info["layout_path"],
                self.fonts_dir,
                target_size=self.config.target_size,
            )

            annotations = renderer.generate_word_annotations(
                data_row,
                image_size,
                add_jitter=self.config.add_bbox_jitter,
            )

            # Step 3: Apply augmentations if enabled
            if self.augmenter and self.config.enable_augmentations:
                # Load image for augmentation
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Apply augmentations
                augmented_image, augmented_annotations = (
                    self.augmenter.apply_augmentations(image_rgb, annotations)
                )

                # Validate augmentation quality
                aug_validation = validate_augmentation_quality(
                    annotations, augmented_annotations
                )
                if not aug_validation["valid"] and self.config.strict_augmentation:
                    logger.warning(
                        f"Augmentation quality issues for {entry_id}: {aug_validation['issues']}"
                    )
                    # Use original if strict mode is enabled
                    augmented_annotations = annotations
                else:
                    # Save augmented image
                    augmented_image_bgr = cv2.cvtColor(
                        augmented_image, cv2.COLOR_RGB2BGR
                    )
                    cv2.imwrite(image_path, augmented_image_bgr)
                    annotations = augmented_annotations

            # Step 4: Validate annotations
            validation_result = renderer.validate_annotations(annotations, image_size)
            if not validation_result["valid"] and self.config.strict_validation:
                return {
                    "success": False,
                    "error": f"Annotation validation failed: {validation_result['issues']}",
                }

            # Step 5: Create XFUND entry
            xfund_entry = renderer.create_xfund_entry(
                entry_id, image_filename, annotations
            )

            # Step 6: Save annotation
            annotation_path = os.path.join(self.annotations_dir, f"{entry_id}.json")
            save_xfund_annotation(xfund_entry, annotation_path)

            # Optional: Generate debug overlay
            if self.config.generate_debug_overlays:
                debug_dir = os.path.join(self.output_dir, "debug")
                ensure_dir_exists(debug_dir)
                debug_path = os.path.join(debug_dir, f"{entry_id}_debug.png")
                renderer.render_debug_overlay(image_path, annotations, debug_path)

            return {
                "success": True,
                "entry_id": entry_id,
                "image_path": image_path,
                "annotation_path": annotation_path,
                "num_annotations": len(annotations),
            }

        except Exception as e:
            logger.error(f"Error generating entry {entry_id}: {e}")
            return {"success": False, "error": str(e)}

    def _generate_single_entry_validated(
        self, data_record: DataRecord, template_info: dict[str, str], index: int
    ) -> dict[str, Any]:
        """
        Generate single dataset entry with Pydantic validation.

        Args:
            data_record: Validated DataRecord model
            template_info: Template information dictionary
            index: Numeric index for the entry

        Returns:
            Dictionary with generation result and validation info
        """
        entry_id = generate_unique_id(index)

        try:
            # Convert DataRecord to dictionary for template processing
            data_dict = data_record.model_dump()

            # Step 1: Fill DOCX template and convert to PNG
            image_filename = f"{entry_id}.png"
            image_path = os.path.join(self.images_dir, image_filename)

            image_path, image_size = process_docx_template(
                template_info["docx_path"],
                data_dict,
                image_path,
                dpi=self.config.image_dpi,
            )

            # Validate image
            if not validate_image_size(image_path):
                return {
                    "success": False,
                    "error": "Generated image too small or invalid",
                }

            # Step 2: Generate word-level annotations with validation
            renderer = WordRenderer(
                template_info["layout_path"],
                self.fonts_dir,
                target_size=self.config.target_size,
            )

            annotations = renderer.generate_word_annotations(
                data_dict, image_size, add_jitter=self.config.add_bbox_jitter
            )

            # Convert to validated XFUNDAnnotation
            try:
                xfund_annotation = self._create_validated_annotation(
                    annotations, image_path, entry_id
                )
            except Exception as e:
                return {"success": False, "error": f"Annotation validation failed: {e}"}

            # Step 3: Apply augmentations if enabled
            if self.augmenter and self.config.enable_augmentations:
                # Load image for augmentation
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Apply augmentations
                augmented_image, augmented_annotations = (
                    self.augmenter.apply_augmentations(image_rgb, annotations)
                )

                # Validate augmentation quality
                aug_validation = validate_augmentation_quality(
                    annotations, augmented_annotations
                )
                if not aug_validation["valid"] and self.config.strict_augmentation:
                    logger.warning(
                        f"Augmentation quality issues for {entry_id}: {aug_validation['issues']}"
                    )
                    # Use original if strict mode is enabled
                    augmented_annotations = annotations
                else:
                    # Save augmented image and update annotations
                    augmented_image_bgr = cv2.cvtColor(
                        augmented_image, cv2.COLOR_RGB2BGR
                    )
                    cv2.imwrite(image_path, augmented_image_bgr)

                    # Update annotation with augmented data
                    try:
                        xfund_annotation = self._create_validated_annotation(
                            augmented_annotations, image_path, entry_id
                        )
                    except Exception as e:
                        logger.warning(f"Augmented annotation validation failed: {e}")
                        # Fall back to original

            # Step 4: Save validated annotation
            annotation_filename = f"{entry_id}.json"
            annotation_path = os.path.join(self.annotations_dir, annotation_filename)
            save_xfund_annotation_pydantic(xfund_annotation, annotation_path)

            # Step 5: Quality validation
            if self.config.strict_validation:
                from .utils import validate_annotation_quality

                quality_issues = validate_annotation_quality(xfund_annotation)
                if quality_issues:
                    logger.warning(f"Quality issues for {entry_id}: {quality_issues}")

            # Generate debug overlays if requested
            if self.config.generate_debug_overlays:
                self._generate_debug_overlay(image_path, xfund_annotation, entry_id)

            return {
                "success": True,
                "entry_id": entry_id,
                "image_path": image_path,
                "annotation_path": annotation_path,
                "entity_count": len(xfund_annotation.form),
            }

        except Exception as e:
            logger.error(f"Error generating entry {entry_id}: {e}")
            return {"success": False, "error": str(e)}

    def _create_validated_annotation(
        self, raw_annotations: list[dict[str, Any]], image_path: str, entry_id: str
    ) -> XFUNDAnnotation:
        """
        Create validated XFUNDAnnotation from raw annotation data.

        Args:
            raw_annotations: List of raw annotation dictionaries
            image_path: Path to the image file
            entry_id: Unique entry identifier

        Returns:
            Validated XFUNDAnnotation object
        """
        from .models import BBoxModel
        from .utils import create_xfund_entity_from_text

        entities = []

        for i, annotation in enumerate(raw_annotations):
            try:
                # Create BBox from coordinates
                bbox = BBoxModel(
                    x1=annotation["bbox"][0],
                    y1=annotation["bbox"][1],
                    x2=annotation["bbox"][2],
                    y2=annotation["bbox"][3],
                )

                # Create validated entity
                entity = create_xfund_entity_from_text(
                    entity_id=i,
                    text=annotation["text"],
                    bbox=bbox,
                    label=annotation.get("label", "OTHER"),
                )

                entities.append(entity)

            except Exception as e:
                logger.warning(f"Failed to create entity {i}: {e}")
                continue

        if not entities:
            raise ValueError("No valid entities created from annotations")

        return XFUNDAnnotation(form=entities, image_path=image_path, image_id=entry_id)

    def _generate_debug_overlay(
        self, image_path: str, annotation: XFUNDAnnotation, entry_id: str
    ) -> None:
        """Generate debug overlay image showing bboxes and text."""
        try:
            import cv2
            from PIL import Image, ImageDraw

            # Load image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Create drawing context
            draw = ImageDraw.Draw(pil_image)

            # Draw bboxes and text
            for entity in annotation.form:
                bbox = entity.bbox

                # Draw bbox rectangle
                draw.rectangle(
                    [bbox.x1, bbox.y1, bbox.x2, bbox.y2], outline="red", width=2
                )

                # Draw entity ID
                draw.text(
                    (bbox.x1, bbox.y1 - 15),
                    f"{entity.id}: {entity.text[:20]}",
                    fill="red",
                )

            # Save debug image
            debug_path = os.path.join(self.images_dir, f"{entry_id}_debug.png")
            pil_image.save(debug_path)

        except Exception as e:
            logger.warning(f"Failed to generate debug overlay: {e}")

    def generate_dataset_with_forms(self) -> GenerationResult:
        """
        Generate XFUND dataset using the form classes for standardized output.

        Returns:
            GenerationResult with form-based XFUND annotations
        """
        start_time = time.time()
        logger.info("Starting XFUND dataset generation using form classes...")

        # Initialize result object
        result = GenerationResult(success=True, total_records=0, templates_used=0)

        try:
            # Load and validate CSV data
            csv_records = load_csv_data_as_models(self.csv_path)
            result.total_records = len(csv_records)
            logger.info(f"Loaded {len(csv_records)} validated records")

            # Generate batch annotations using form classes
            generation_results = self.form_generator.generate_batch_xfund_annotations(
                templates_dir=self.templates_dir,
                data_records=csv_records,
                output_dir=self.output_dir,
            )

            # Process results
            successful = [r for r in generation_results if r["status"] == "success"]
            failed = [r for r in generation_results if r["status"] == "failed"]

            result.generated_entries = len(successful)
            result.templates_used = len({r["template_name"] for r in successful})

            # Add errors from failed generations
            for failure in failed:
                result.add_error(
                    f"Template {failure['template_name']}: {failure.get('error', 'Unknown error')}"
                )

            # Log results
            logger.info("Form-based generation completed:")
            logger.info(f"  Success: {len(successful)} entries")
            logger.info(f"  Failed: {len(failed)} entries")
            logger.info(f"  Templates used: {result.templates_used}")

            # Log Q&A statistics
            total_qa_pairs = sum(r.get("qa_pairs", 0) for r in successful)
            logger.info(f"  Total Q&A pairs generated: {total_qa_pairs}")

        except Exception as e:
            logger.error(f"Form-based generation failed: {e}")
            result.success = False
            result.add_error(str(e))

        # Calculate timing
        result.generation_time = time.time() - start_time
        result.output_paths = {
            "images": self.images_dir,
            "annotations": self.annotations_dir,
        }

        return result

    def validate_setup(self) -> dict[str, Any]:
        """
        Validate generator setup and dependencies.

        Returns:
            Validation results
        """
        issues = []

        # Check LibreOffice installation
        if not check_libreoffice_installed():
            issues.append("LibreOffice not installed or not in PATH")

        # Check input paths
        if not os.path.exists(self.templates_dir):
            issues.append(f"Templates directory not found: {self.templates_dir}")

        if not os.path.exists(self.csv_path):
            issues.append(f"CSV file not found: {self.csv_path}")

        # Check templates
        templates = self._find_templates()
        if not templates:
            issues.append("No valid templates found")

        # Check CSV data
        try:
            csv_data = load_csv_data(self.csv_path)
            if not csv_data:
                issues.append("CSV file is empty")
            elif len(csv_data[0]) < 3:  # Expect at least a few columns
                issues.append("CSV file has very few columns")
        except Exception as e:
            issues.append(f"Failed to load CSV: {e}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "templates_found": len(templates) if "templates" in locals() else 0,
        }


def create_default_config() -> dict[str, Any]:
    """Create default configuration for dataset generation."""
    return {
        "templates_dir": "data/templates_docx",
        "csv_path": "data/csv/data.csv",
        "output_dir": "output",
        "fonts_dir": "fonts/handwritten_fonts",
        "image_dpi": 300,
        "target_size": 1000,
        "enable_augmentations": True,
        "augmentation_difficulty": "medium",
        "document_type": "medical",
        "add_bbox_jitter": True,
        "strict_validation": False,
        "strict_augmentation": False,
        "generate_debug_overlays": False,
    }


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate XFUND-style OCR dataset")

    parser.add_argument(
        "--config", "-c", type=str, help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--templates-dir",
        "-t",
        type=str,
        default="data/templates_docx",
        help="Directory containing DOCX templates",
    )
    parser.add_argument(
        "--csv-path",
        "-d",
        type=str,
        default="data/csv/data.csv",
        help="Path to CSV data file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output",
        help="Output directory for generated dataset",
    )
    parser.add_argument(
        "--no-augmentations", action="store_true", help="Disable image augmentations"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Generate debug overlay images"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate setup, don't generate dataset",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = create_default_config()
        logger.info("Using default configuration")

    # Override config with command line arguments
    if args.templates_dir:
        config["templates_dir"] = args.templates_dir
    if args.csv_path:
        config["csv_path"] = args.csv_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.no_augmentations:
        config["enable_augmentations"] = False
    if args.debug:
        config["generate_debug_overlays"] = True

    # Convert relative paths to absolute
    for path_key in ["templates_dir", "csv_path", "output_dir"]:
        if path_key in config and not os.path.isabs(config[path_key]):
            config[path_key] = os.path.abspath(config[path_key])

    # Initialize generator with validated config
    try:
        generator_config = GeneratorConfig(**config)
        generator = XFUNDGenerator(generator_config)
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        return 1

    # Validate setup
    validation_result = generator.validate_setup()
    if not validation_result["valid"]:
        logger.error("Setup validation failed:")
        for issue in validation_result["issues"]:
            logger.error(f"  - {issue}")
        return 1

    logger.info("Setup validation passed")
    logger.info(f"Found {validation_result['templates_found']} template(s)")

    if args.validate_only:
        logger.info("Validation-only mode, exiting")
        return 0

    # Generate dataset
    try:
        results = generator.generate_dataset()

        if results["success"]:
            logger.info("Dataset generation completed successfully!")
            logger.info(f"Generated {results['generated_entries']} entries")

            # Save generation report
            report_path = os.path.join(config["output_dir"], "generation_report.json")
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Generation report saved to {report_path}")

            return 0
        else:
            logger.error("Dataset generation failed")
            return 1

    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during generation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
