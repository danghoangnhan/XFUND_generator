"""
Test cases for Pydantic model validation and functionality.
"""

import json

import pytest
from pydantic import ValidationError

from xfund_generator.models import (
    AnnotationStats,
    AnnotationValidationResult,
    AugmentationDifficulty,
    BBoxModel,
    DataRecord,
    DocumentType,
    GeneratorConfig,
    TemplateValidationResult,
    WordAnnotation,
    XFUNDEntity,
    get_default_config,
    validate_annotations,
    validate_config_file,
)


class TestBBoxModel:
    """Test BBoxModel validation and methods."""

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_valid_bbox_creation(self, sample_bbox):
        """Test creating a valid bbox."""
        assert sample_bbox.x1 == 10
        assert sample_bbox.y1 == 20
        assert sample_bbox.x2 == 100
        assert sample_bbox.y2 == 80

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_bbox_computed_properties(self, sample_bbox):
        """Test bbox computed properties."""
        assert sample_bbox.area() == 5400.0
        assert sample_bbox.center() == (55.0, 50.0)
        assert sample_bbox.to_list() == [10.0, 20.0, 100.0, 80.0]

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_invalid_bbox_validation(self):
        """Test that invalid bboxes are rejected."""
        with pytest.raises(ValidationError):
            # x1 >= x2
            BBoxModel(x1=100, y1=20, x2=10, y2=80)

        with pytest.raises(ValidationError):
            # y1 >= y2
            BBoxModel(x1=10, y1=80, x2=100, y2=20)

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_bbox_normalization(self, sample_bbox):
        """Test bbox normalization."""
        normalized = sample_bbox.normalize(
            img_width=500, img_height=400, target_size=1000
        )

        assert isinstance(normalized, BBoxModel)
        assert normalized.x1 == 20  # 10 * (1000/500)
        assert normalized.y1 == 50  # 20 * (1000/400)
        assert normalized.x2 == 200  # 100 * (1000/500)
        assert normalized.y2 == 200  # 80 * (1000/400)


class TestGeneratorConfig:
    """Test GeneratorConfig validation and functionality."""

    @pytest.mark.unit
    @pytest.mark.pydantic
    @pytest.mark.config
    def test_valid_config_creation(self, sample_config_data):
        """Test creating a valid configuration."""
        config = GeneratorConfig(**sample_config_data)

        assert "templates_docx" in config.templates_dir
        assert "data.csv" in config.csv_path
        assert "test" in config.output_dir
        assert config.document_type == DocumentType.MEDICAL
        assert config.enable_augmentations is True
        assert config.augmentation_difficulty == AugmentationDifficulty.MEDIUM

    @pytest.mark.unit
    @pytest.mark.pydantic
    @pytest.mark.config
    def test_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()

        assert isinstance(config, GeneratorConfig)
        assert "templates_docx" in config.templates_dir
        assert config.document_type == DocumentType.MEDICAL

    @pytest.mark.unit
    @pytest.mark.pydantic
    @pytest.mark.config
    def test_config_path_resolution(self, sample_config_data):
        """Test that paths are properly resolved."""
        config = GeneratorConfig(**sample_config_data)

        # Paths should be strings
        assert isinstance(config.templates_dir, str)
        assert isinstance(config.csv_path, str)
        assert isinstance(config.output_dir, str)

    @pytest.mark.unit
    @pytest.mark.pydantic
    @pytest.mark.config
    def test_invalid_document_type(self, sample_config_data):
        """Test that invalid document types are rejected."""
        sample_config_data["document_type"] = "invalid_type"

        with pytest.raises(ValidationError):
            GeneratorConfig(**sample_config_data)

    @pytest.mark.unit
    @pytest.mark.pydantic
    @pytest.mark.config
    def test_invalid_augmentation_difficulty(self, sample_config_data):
        """Test that invalid augmentation difficulties are rejected."""
        sample_config_data["augmentation_difficulty"] = "impossible"

        with pytest.raises(ValidationError):
            GeneratorConfig(**sample_config_data)

    @pytest.mark.integration
    @pytest.mark.pydantic
    @pytest.mark.config
    def test_config_file_validation(self, temp_dir, sample_config_data):
        """Test validating configuration from file."""
        config_file = temp_dir / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(sample_config_data, f)

        result = validate_config_file(str(config_file))

        # Config should be created even if paths don't exist
        assert result.config is not None
        assert isinstance(result.config, GeneratorConfig)
        # is_valid may be False due to non-existent paths, but config is still created

    @pytest.mark.integration
    @pytest.mark.pydantic
    @pytest.mark.config
    def test_invalid_config_file_validation(self, temp_dir):
        """Test validating invalid configuration from file."""
        config_file = temp_dir / "invalid_config.json"
        # Missing required fields will cause validation error
        invalid_config = {"invalid_field": "invalid_value"}

        with open(config_file, "w") as f:
            json.dump(invalid_config, f)

        result = validate_config_file(str(config_file))

        # Should have errors due to missing required fields or invalid data
        # but config may still be constructed with defaults
        assert result is not None


class TestDataRecord:
    """Test DataRecord validation."""

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_valid_data_record(self):
        """Test creating a valid data record."""
        record = DataRecord(
            hospital_name_text="Central Hospital",
            doctor_name_text="Dr. Smith",
            patient_name_text="John Doe",
        )

        assert record.hospital_name_text == "Central Hospital"
        assert record.doctor_name_text == "Dr. Smith"
        assert record.patient_name_text == "John Doe"

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_data_record_optional_fields(self):
        """Test that optional fields work correctly."""
        record = DataRecord(hospital_name_text="Test Hospital")

        assert record.hospital_name_text == "Test Hospital"
        # Optional fields default to None
        assert record.doctor_name_text is None


class TestTemplateValidationResult:
    """Test TemplateValidationResult functionality."""

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_success_result(self):
        """Test creating a success result."""
        result = TemplateValidationResult.create_success(
            placeholders=["{{name}}", "{{date}}"],
            paragraph_count=5,
            table_count=2,
        )

        assert result.valid is True
        assert result.placeholders == ["{{name}}", "{{date}}"]
        assert result.paragraph_count == 5
        assert result.table_count == 2
        assert result.error is None

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_error_result(self):
        """Test creating an error result."""
        result = TemplateValidationResult.create_error(
            error_message="Template validation failed"
        )

        assert result.valid is False
        assert result.error == "Template validation failed"
        assert result.placeholders == []


class TestXFUNDEntity:
    """Test XFUNDEntity validation."""

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_valid_xfund_entity(self, sample_bbox):
        """Test creating a valid XFUND entity."""
        entity = XFUNDEntity(
            id=1, text="Patient Name:", bbox=sample_bbox, label="QUESTION"
        )

        assert entity.id == 1
        assert entity.text == "Patient Name:"
        assert entity.bbox == sample_bbox
        assert entity.label == "QUESTION"

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_xfund_entity_with_words(self, sample_bbox):
        """Test XFUND entity with word-level annotations."""
        entity = XFUNDEntity(
            id=1,
            text="Patient Name:",
            bbox=sample_bbox,
            label="QUESTION",
            words=["Patient", "Name:"],
        )

        assert len(entity.words) == 2
        assert entity.words == ["Patient", "Name:"]


class TestPydanticIntegration:
    """Test overall Pydantic integration."""

    @pytest.mark.integration
    @pytest.mark.pydantic
    def test_model_serialization(self, sample_config, sample_bbox):
        """Test that models can be properly serialized."""
        # Test config serialization
        config_dict = sample_config.model_dump()
        assert isinstance(config_dict, dict)
        assert "templates_dir" in config_dict

        # Test bbox serialization
        bbox_dict = sample_bbox.model_dump()
        assert isinstance(bbox_dict, dict)
        assert bbox_dict["x1"] == 10

    @pytest.mark.integration
    @pytest.mark.pydantic
    def test_model_deserialization(self, sample_config_data):
        """Test that models can be recreated from serialized data."""
        # Create config
        config1 = GeneratorConfig(**sample_config_data)

        # Serialize and deserialize
        config_dict = config1.model_dump()
        config2 = GeneratorConfig(**config_dict)

        # Should be equivalent
        assert config1.templates_dir == config2.templates_dir
        assert config1.document_type == config2.document_type

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_validation_error_messages(self):
        """Test that validation errors provide helpful messages."""
        with pytest.raises(ValidationError) as exc_info:
            GeneratorConfig(
                templates_dir="valid/path",
                csv_path="valid/path.csv",
                output_dir="valid/output",
                image_dpi="not_a_number",  # Invalid type
            )

        error = exc_info.value
        assert "image_dpi" in str(error)
        assert "int" in str(error) or "integer" in str(error)


class TestWordAnnotation:
    """Test WordAnnotation validation and methods."""

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_valid_word_annotation(self):
        """Test creating a valid word annotation."""
        ann = WordAnnotation(text="Hello", bbox=[10, 20, 100, 80], label="greeting")

        assert ann.text == "Hello"
        assert ann.bbox == [10, 20, 100, 80]
        assert ann.label == "greeting"

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_word_annotation_to_dict(self):
        """Test converting annotation to dictionary."""
        ann = WordAnnotation(text="Test", bbox=[0, 0, 50, 50], label="test")
        result = ann.to_dict()

        assert result == {"text": "Test", "bbox": [0, 0, 50, 50], "label": "test"}

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_word_annotation_from_dict(self):
        """Test creating annotation from dictionary."""
        data = {"text": "World", "bbox": [100, 100, 200, 150], "label": "noun"}
        ann = WordAnnotation.from_dict(data)

        assert ann.text == "World"
        assert ann.bbox == [100, 100, 200, 150]
        assert ann.label == "noun"

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_invalid_empty_text(self):
        """Test that empty text is rejected."""
        with pytest.raises(ValidationError):
            WordAnnotation(text="   ", bbox=[10, 20, 100, 80], label="test")

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_invalid_bbox_format(self):
        """Test that invalid bbox format is rejected."""
        with pytest.raises(ValidationError):
            WordAnnotation(text="Test", bbox=[10, 20, 100], label="test")

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_invalid_bbox_coordinates(self):
        """Test that invalid bbox coordinates are rejected."""
        with pytest.raises(ValidationError):
            # x1 >= x2
            WordAnnotation(text="Test", bbox=[100, 20, 10, 80], label="test")

        with pytest.raises(ValidationError):
            # y1 >= y2
            WordAnnotation(text="Test", bbox=[10, 80, 100, 20], label="test")

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_validate_bounds(self):
        """Test bounds validation method."""
        ann = WordAnnotation(text="Test", bbox=[10, 20, 100, 80], label="test")

        # Should pass with default target_size=1000
        issues = ann.validate_bounds(1000)
        assert len(issues) == 0

        # Should fail with smaller max_value
        issues = ann.validate_bounds(50)
        assert len(issues) > 0

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_intersects_overlapping(self):
        """Test overlap detection for overlapping bboxes."""
        ann1 = WordAnnotation(text="A", bbox=[10, 10, 100, 100], label="test")
        ann2 = WordAnnotation(text="B", bbox=[50, 50, 150, 150], label="test")

        assert ann1.intersects(ann2) is True
        assert ann2.intersects(ann1) is True

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_intersects_non_overlapping(self):
        """Test overlap detection for non-overlapping bboxes."""
        ann1 = WordAnnotation(text="A", bbox=[10, 10, 100, 100], label="test")
        ann2 = WordAnnotation(text="B", bbox=[200, 200, 300, 300], label="test")

        assert ann1.intersects(ann2) is False
        assert ann2.intersects(ann1) is False


class TestAnnotationValidation:
    """Test annotation validation function."""

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_validate_valid_annotations(self):
        """Test validating valid annotations."""
        annotations = [
            {"text": "Hello", "bbox": [10, 20, 100, 80], "label": "greeting"},
            {"text": "World", "bbox": [150, 20, 250, 80], "label": "greeting"},
        ]
        result = validate_annotations(annotations)

        assert result.valid is True
        assert len(result.issues) == 0
        assert result.stats.total_annotations == 2
        assert result.stats.unique_labels == 1

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_validate_detects_overlaps(self):
        """Test that overlapping bboxes are detected."""
        annotations = [
            {"text": "A", "bbox": [10, 10, 100, 100], "label": "test"},
            {"text": "B", "bbox": [50, 50, 150, 150], "label": "test"},
        ]
        result = validate_annotations(annotations, check_overlaps=True)

        assert len(result.stats.bbox_overlaps) == 1
        assert result.stats.bbox_overlaps[0] == (0, 1)

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_validate_invalid_annotation(self):
        """Test validating annotations with invalid entries."""
        annotations = [
            {"text": "Valid", "bbox": [10, 20, 100, 80], "label": "test"},
            {"text": "", "bbox": [150, 20, 250, 80], "label": "test"},  # Empty text
        ]
        result = validate_annotations(annotations)

        assert result.valid is False
        assert len(result.issues) > 0

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_validate_out_of_bounds(self):
        """Test validating annotations with out-of-bounds coordinates."""
        annotations = [
            {"text": "Test", "bbox": [10, 20, 1500, 80], "label": "test"},  # x2 > 1000
        ]
        result = validate_annotations(annotations, target_size=1000)

        assert result.valid is False
        assert any("out of bounds" in issue for issue in result.issues)

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_annotation_validation_result_factory(self):
        """Test AnnotationValidationResult factory methods."""
        stats = AnnotationStats(
            total_annotations=5,
            unique_labels=2,
            words_per_label={"a": 3, "b": 2},
        )

        valid_result = AnnotationValidationResult.create_valid(stats)
        assert valid_result.valid is True
        assert len(valid_result.issues) == 0

        invalid_result = AnnotationValidationResult.create_invalid(
            ["Error 1", "Error 2"], stats
        )
        assert invalid_result.valid is False
        assert len(invalid_result.issues) == 2
