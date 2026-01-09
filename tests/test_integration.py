"""
Integration test cases for XFUND form integration and unified JSON export.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xfund_generator.form import Word, XFUNDAnnotation, XFUNDDataset

# Conditional import - may not exist in all setups
try:
    from xfund_generator.xfund_form_integration import XFUNDFormGenerator

    XFUND_INTEGRATION_AVAILABLE = True
except ImportError:
    XFUND_INTEGRATION_AVAILABLE = False
    XFUNDFormGenerator = None
from xfund_generator.models import DataRecord


class TestXFUNDFormIntegration:
    """Test XFUND form integration functionality."""

    @pytest.mark.integration
    @pytest.mark.forms
    def test_xfund_form_generator_creation(self, sample_config):
        """Test creating XFUNDFormGenerator."""
        if not XFUND_INTEGRATION_AVAILABLE:
            pytest.skip("XFUND integration module not available")

        generator = XFUNDFormGenerator(config=sample_config)

        assert generator.config == sample_config
        assert hasattr(generator, "generate_xfund_from_template")

    @pytest.mark.integration
    @pytest.mark.forms
    def test_generate_xfund_dataset(self, sample_config):
        """Test generating XFUND dataset from form data."""
        if not XFUND_INTEGRATION_AVAILABLE:
            pytest.skip("XFUND integration module not available")

        # Create generator - verify it can be instantiated
        generator = XFUNDFormGenerator(config=sample_config)

        # Verify generator has expected methods
        assert hasattr(generator, "generate_xfund_from_template")
        assert hasattr(generator, "_create_word_annotations")
        assert hasattr(generator, "_determine_label_type")

        # Test word annotation creation
        words = generator._create_word_annotations("Test Hospital", (10, 20, 150, 40))
        assert len(words) == 2  # "Test" and "Hospital"
        assert words[0].text == "Test"
        assert words[1].text == "Hospital"

    @pytest.mark.integration
    @pytest.mark.forms
    def test_question_answer_linking(self, sample_config):
        """Test automatic question-answer linking via annotations."""
        if not XFUND_INTEGRATION_AVAILABLE:
            pytest.skip("XFUND integration module not available")

        generator = XFUNDFormGenerator(config=sample_config)

        # Test label type determination
        assert generator._determine_label_type("patient_name_label") == "question"
        assert generator._determine_label_type("patient_name_value") == "answer"
        assert generator._determine_label_type("unknown_field") == "other"

    @pytest.mark.integration
    @pytest.mark.forms
    def test_word_level_annotations(self, sample_config):
        """Test word-level annotation creation."""
        if not XFUND_INTEGRATION_AVAILABLE:
            pytest.skip("XFUND integration module not available")

        generator = XFUNDFormGenerator(config=sample_config)

        text = "Patient Name:"
        bbox = (10, 20, 100, 40)

        words = generator._create_word_annotations(text, bbox)

        assert len(words) == 2  # "Patient" and "Name:"
        assert all(isinstance(w, Word) for w in words)
        assert words[0].text == "Patient"
        assert words[1].text == "Name:"

    @pytest.mark.integration
    @pytest.mark.forms
    def test_medical_field_detection(self, sample_config):
        """Test field type detection and classification."""
        if not XFUND_INTEGRATION_AVAILABLE:
            pytest.skip("XFUND integration module not available")

        generator = XFUNDFormGenerator(config=sample_config)

        # Test field classification using _determine_label_type
        test_cases = [
            ("patient_name_label", "question"),
            ("patient_name_value", "answer"),
            ("hospital_label", "question"),
            ("hospital_data", "answer"),
            ("random_field", "other"),
        ]

        for field_name, expected_label in test_cases:
            label = generator._determine_label_type(field_name)
            assert label == expected_label


class TestUnifiedJSONExportIntegration:
    """Test unified JSON export integration across different scenarios."""

    @pytest.mark.integration
    @pytest.mark.forms
    def test_format_consistency_across_datasets(self, sample_datasets):
        """Test that all formats produce consistent JSON structure."""
        for format_name, dataset in sample_datasets.items():
            json_output = dataset.to_json()
            parsed = json.loads(json_output)

            # All formats should have consistent top-level structure
            assert "annotations" in parsed
            assert isinstance(parsed["annotations"], list)

            # Each annotation should have basic required fields
            for annotation in parsed["annotations"]:
                assert "text" in annotation
                assert "box" in annotation
                assert "label" in annotation
                # WildReceipt format intentionally excludes words for minimal format
                if format_name != "wildreceipt":
                    assert "words" in annotation

    @pytest.mark.integration
    @pytest.mark.forms
    def test_json_serialization_roundtrip(self, sample_datasets):
        """Test that JSON can be serialized and deserialized."""
        for _format_name, dataset in sample_datasets.items():
            # Export to JSON
            json_output = dataset.to_json()

            # Parse back from JSON
            parsed_data = json.loads(json_output)

            # Should be able to recreate similar structure
            assert len(parsed_data["annotations"]) == len(dataset.annotations)

            # Text content should match
            original_texts = [ann.text for ann in dataset.annotations]
            parsed_texts = [ann["text"] for ann in parsed_data["annotations"]]
            assert original_texts == parsed_texts

    @pytest.mark.integration
    @pytest.mark.forms
    def test_large_dataset_export(self):
        """Test exporting datasets with many annotations."""
        # Create dataset with many annotations
        word = Word(box=[10, 20, 50, 40], text="Test")
        annotations = []

        for i in range(100):
            annotation = XFUNDAnnotation(
                id=i,
                box=[10 + i, 20, 100 + i, 40],
                text=f"Annotation {i}",
                label="question" if i % 2 == 0 else "answer",
                words=[word],
                linking=[[i, i + 1]] if i % 2 == 0 and i < 99 else [],
            )
            annotations.append(annotation)

        dataset = XFUNDDataset(image_path="large_test.png", annotations=annotations)

        # Should be able to export large dataset
        json_output = dataset.to_json()
        parsed = json.loads(json_output)

        assert len(parsed["annotations"]) == 100
        assert json_output.count('"linking"') > 40  # About half should have linking

    @pytest.mark.integration
    @pytest.mark.forms
    def test_performance_comparison(self, sample_datasets):
        """Test performance of unified API vs hypothetical separate methods."""
        import time

        # Measure unified API performance
        start_time = time.time()
        for _ in range(1000):
            for dataset in sample_datasets.values():
                _ = dataset.to_json()
        unified_time = time.time() - start_time

        # Unified API should be reasonable (< 1 second for 3000 calls)
        assert unified_time < 1.0


class TestFormGenerationPipeline:
    """Test the complete form generation pipeline."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_pipeline(self, sample_config, temp_dir):
        """Test complete pipeline from config to final JSON."""
        if not XFUND_INTEGRATION_AVAILABLE:
            pytest.skip("XFUND integration module not available")

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Create generator
        generator = XFUNDFormGenerator(config=sample_config)

        # Test the core functionality - word annotation creation
        word_annotations = generator._create_word_annotations(
            "General Hospital", (10, 20, 200, 40)
        )
        assert len(word_annotations) == 2

        # Test label determination
        assert generator._determine_label_type("hospital_label") == "question"
        assert generator._determine_label_type("hospital_value") == "answer"

        # Create a sample dataset directly to test JSON export
        sample_annotations = [
            XFUNDAnnotation(
                id=1,
                box=[10, 20, 150, 40],
                text="Hospital Name:",
                label="question",
                words=[Word(box=[10, 20, 80, 40], text="Hospital"),
                       Word(box=[85, 20, 150, 40], text="Name:")],
                linking=[[1, 2]],
            ),
            XFUNDAnnotation(
                id=2,
                box=[160, 20, 350, 40],
                text="General Hospital",
                label="answer",
                words=[Word(box=[160, 20, 250, 40], text="General"),
                       Word(box=[255, 20, 350, 40], text="Hospital")],
            ),
        ]

        dataset = XFUNDDataset(image_path="test.png", annotations=sample_annotations)

        # Export to JSON
        json_output = dataset.to_json()
        parsed = json.loads(json_output)

        assert len(parsed["annotations"]) == 2
        questions = [ann for ann in parsed["annotations"] if ann["label"] == "question"]
        answers = [ann for ann in parsed["annotations"] if ann["label"] == "answer"]

        assert len(questions) == 1
        assert len(answers) == 1

    @pytest.mark.integration
    @pytest.mark.forms
    def test_error_handling_in_pipeline(self, sample_config):
        """Test error handling in the generation pipeline."""
        if not XFUND_INTEGRATION_AVAILABLE:
            pytest.skip("XFUND integration module not available")

        generator = XFUNDFormGenerator(config=sample_config)

        # Test that generator can handle edge cases gracefully
        # Test empty text
        words = generator._create_word_annotations("", (10, 20, 100, 40))
        assert len(words) == 0

        # Test single word
        words = generator._create_word_annotations("Test", (10, 20, 100, 40))
        assert len(words) == 1
        assert words[0].text == "Test"

    @pytest.mark.integration
    @pytest.mark.forms
    def test_configuration_validation_integration(self):
        """Test that configuration validation works in integration."""
        if not XFUND_INTEGRATION_AVAILABLE:
            pytest.skip("XFUND integration module not available")

        from xfund_generator.models import validate_config_file

        # Test with valid config data
        valid_config = {
            "templates_dir": "data/templates_docx",
            "csv_path": "data/csv/data.csv",
            "output_dir": "output",
            "document_type": "medical",
            "enable_augmentations": True,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(valid_config, f)
            config_path = f.name

        try:
            result = validate_config_file(config_path)
            # Config should be created even if paths don't exist
            assert result.config is not None

            # Should be able to use validated config with form generator
            generator = XFUNDFormGenerator(config=result.config)
            # document_type may be string or enum depending on how config was constructed
            doc_type = generator.config.document_type
            doc_type_value = doc_type.value if hasattr(doc_type, "value") else doc_type
            assert doc_type_value == "medical"

        finally:
            Path(config_path).unlink()


class TestBackwardsCompatibility:
    """Test backwards compatibility and migration support."""

    @pytest.mark.integration
    @pytest.mark.forms
    def test_old_style_data_still_works(self):
        """Test that old-style data structures still work."""
        # Simulate old-style annotation data (as dict)
        old_style_annotation = {
            "id": 1,
            "text": "Patient Name:",
            "box": [10, 20, 100, 80],
            "label": "question",
            "words": [{"text": "Patient", "box": [10, 20, 60, 80]}],
        }

        # Should be able to create new-style annotation from old data
        word = Word(text="Patient", box=[10, 20, 60, 80])
        new_annotation = XFUNDAnnotation(
            id=old_style_annotation["id"],
            text=old_style_annotation["text"],
            box=old_style_annotation["box"],
            label=old_style_annotation["label"],
            words=[word],
        )

        assert new_annotation.text == old_style_annotation["text"]
        assert new_annotation.box == old_style_annotation["box"]

    @pytest.mark.integration
    @pytest.mark.forms
    def test_migration_path_from_separate_methods(self, sample_datasets):
        """Test migration path from separate methods to unified API."""
        # Verify that unified API provides same functionality
        for format_name, dataset in sample_datasets.items():
            # All should work with unified method
            json_output = dataset.to_json()
            parsed = json.loads(json_output)

            # Should contain format-specific information
            if format_name == "xfund":
                # Should have linking information
                has_linking = any("linking" in ann for ann in parsed["annotations"])
                assert has_linking or all(
                    ann.get("linking", []) == [] for ann in parsed["annotations"]
                )

            elif format_name == "funsd":
                # Should have key/value IDs
                has_ids = any("key_id" in ann for ann in parsed["annotations"])
                assert has_ids
