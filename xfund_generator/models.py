"""
Pydantic models for XFUND dataset generation.
Provides data validation, serialization, and type safety.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DocumentType(str, Enum):
    """Supported document types for generation."""

    MEDICAL = "medical"
    FORM = "form"
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    GENERAL = "general"


class AugmentationDifficulty(str, Enum):
    """Augmentation difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


class BBoxModel(BaseModel):
    """Pydantic model for bounding box coordinates."""

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")

    @field_validator("x1", "x2")
    @classmethod
    def validate_x_coords(cls, v):
        if v < 0:
            raise ValueError("X coordinates must be non-negative")
        return v

    @field_validator("y1", "y2")
    @classmethod
    def validate_y_coords(cls, v):
        if v < 0:
            raise ValueError("Y coordinates must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_bbox_order(self):
        if self.x1 >= self.x2:
            raise ValueError("x1 must be less than x2")
        if self.y1 >= self.y2:
            raise ValueError("y1 must be less than y2")
        return self

    def to_list(self) -> list[float]:
        """Convert to list format [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]

    def to_xfund_format(self) -> list[int]:
        """Convert to XFUND integer format [x1, y1, x2, y2]."""
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]

    def normalize(
        self, img_width: int, img_height: int, target_size: int = 1000
    ) -> "BBoxModel":
        """Normalize bbox to 0-target_size scale."""
        return BBoxModel(
            x1=(self.x1 / img_width) * target_size,
            y1=(self.y1 / img_height) * target_size,
            x2=(self.x2 / img_width) * target_size,
            y2=(self.y2 / img_height) * target_size,
        )

    def area(self) -> float:
        """Calculate bbox area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def center(self) -> tuple[float, float]:
        """Calculate bbox center point."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class XFUNDEntity(BaseModel):
    """XFUND entity with text and bounding box."""

    model_config = ConfigDict(validate_assignment=True)

    id: int = Field(..., description="Entity ID")
    text: str = Field(..., description="Entity text content")
    bbox: BBoxModel = Field(..., description="Bounding box coordinates")
    label: str = Field(..., description="Entity label/category")
    words: Optional[list[str]] = Field(default=None, description="Individual words")
    linking: Optional[list[list[int]]] = Field(
        default=None, description="Entity linking information"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Entity text cannot be empty")
        return v.strip()

    @field_validator("label")
    @classmethod
    def validate_label(cls, v):
        allowed_labels = {"HEADER", "QUESTION", "ANSWER", "OTHER"}
        if v.upper() not in allowed_labels:
            raise ValueError(f"Label must be one of {allowed_labels}")
        return v.upper()


class XFUNDAnnotation(BaseModel):
    """Complete XFUND annotation for a document."""

    model_config = ConfigDict(validate_assignment=True)

    form: list[XFUNDEntity] = Field(..., description="List of entities in the document")
    image_path: str = Field(..., description="Path to the source image")
    image_id: Optional[str] = Field(default=None, description="Unique image identifier")

    @field_validator("form")
    @classmethod
    def validate_entities(cls, v):
        if not v:
            raise ValueError("Document must contain at least one entity")
        return v


class AugmentationConfig(BaseModel):
    """Configuration for document augmentations."""

    model_config = ConfigDict(validate_assignment=True)

    difficulty: AugmentationDifficulty = Field(default=AugmentationDifficulty.MEDIUM)
    brightness_range: tuple[float, float] = Field(
        default=(0.8, 1.2), description="Brightness adjustment range"
    )
    contrast_range: tuple[float, float] = Field(
        default=(0.8, 1.2), description="Contrast adjustment range"
    )
    blur_probability: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Probability of applying blur"
    )
    noise_probability: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Probability of adding noise"
    )
    rotation_range: tuple[float, float] = Field(
        default=(-2.0, 2.0), description="Rotation angle range in degrees"
    )
    perspective_probability: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Probability of perspective transform"
    )

    @field_validator("brightness_range", "contrast_range")
    @classmethod
    def validate_ranges(cls, v):
        if v[0] >= v[1]:
            raise ValueError("Range minimum must be less than maximum")
        if v[0] <= 0:
            raise ValueError("Range values must be positive")
        return v


class TemplateConfig(BaseModel):
    """Configuration for document template."""

    model_config = ConfigDict(validate_assignment=True)

    template_path: str = Field(..., description="Path to template file")
    template_type: str = Field(default="docx", description="Template file type")
    fields: dict[str, Any] = Field(
        default_factory=dict, description="Template field mappings"
    )
    layout_config: Optional[dict[str, Any]] = Field(
        default=None, description="Layout configuration"
    )

    @field_validator("template_path")
    @classmethod
    def validate_template_path(cls, v):
        """Validate template path - allows non-existent paths for testing."""
        return str(Path(v))


class DataRecord(BaseModel):
    """Single data record from CSV input."""

    model_config = ConfigDict(validate_assignment=True, extra="allow")

    # Medical document fields
    hospital_name_text: Optional[str] = Field(default=None, description="Hospital name")
    hospital_address_text: Optional[str] = Field(
        default=None, description="Hospital address"
    )
    doctor_name_text: Optional[str] = Field(default=None, description="Doctor name")
    patient_name_text: Optional[str] = Field(default=None, description="Patient name")
    department_text: Optional[str] = Field(default=None, description="Medical department")
    diagnose_text: Optional[str] = Field(default=None, description="Diagnosis")
    doctor_comment_text: Optional[str] = Field(
        default=None, description="Doctor comments"
    )

    # Additional fields for other document types
    additional_fields: Optional[dict[str, str]] = Field(
        default_factory=dict, description="Additional custom fields"
    )

    def get_field(self, field_name: str) -> str:
        """Get field value by name with fallback to additional_fields."""
        if hasattr(self, field_name):
            return getattr(self, field_name) or ""
        return self.additional_fields.get(field_name, "")


class GeneratorConfig(BaseModel):
    """Main configuration for XFUND dataset generation."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",  # Allow additional fields for flexibility
    )

    # Required paths
    templates_dir: str = Field(..., description="Directory containing templates")
    csv_path: str = Field(..., description="Path to CSV data file")
    output_dir: str = Field(..., description="Output directory for generated dataset")

    # Optional paths
    fonts_dir: Optional[str] = Field(
        default=None, description="Directory containing custom fonts"
    )

    # Image settings
    image_dpi: int = Field(
        default=300, ge=72, le=600, description="Image resolution in DPI"
    )
    target_size: int = Field(
        default=1000, ge=224, description="Target size for XFUND normalization"
    )

    # Generation settings
    document_type: DocumentType = Field(
        default=DocumentType.MEDICAL, description="Type of documents to generate"
    )
    enable_augmentations: bool = Field(
        default=True, description="Enable data augmentations"
    )
    augmentation_difficulty: AugmentationDifficulty = Field(
        default=AugmentationDifficulty.MEDIUM
    )

    # Quality settings
    add_bbox_jitter: bool = Field(
        default=True, description="Add small random jitter to bounding boxes"
    )
    strict_validation: bool = Field(
        default=False, description="Enable strict validation"
    )
    strict_augmentation: bool = Field(
        default=False, description="Enable strict augmentation validation"
    )
    generate_debug_overlays: bool = Field(
        default=False, description="Generate debug overlay images"
    )

    # Performance settings
    max_workers: int = Field(
        default=4, ge=1, description="Maximum number of worker processes"
    )
    batch_size: int = Field(default=10, ge=1, description="Batch size for processing")

    @field_validator("templates_dir", "csv_path")
    @classmethod
    def validate_paths(cls, v):
        """Validate and normalize path strings."""
        path = Path(v)
        if path.exists():
            return str(path.resolve())
        return str(path)

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v):
        path = Path(v)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return str(path.resolve())
        except (OSError, PermissionError):
            return str(path)

    @classmethod
    def from_json_file(cls, file_path: str) -> "GeneratorConfig":
        """Load configuration from JSON file."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json_file(self, file_path: str) -> None:
        """Save configuration to JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    def get_augmentation_config(self) -> AugmentationConfig:
        """Get augmentation configuration based on current settings."""
        return AugmentationConfig(difficulty=self.augmentation_difficulty)


class GenerationResult(BaseModel):
    """Result of dataset generation process."""

    model_config = ConfigDict(validate_assignment=True)

    success: bool = Field(..., description="Whether generation was successful")
    total_records: int = Field(default=0, description="Total number of input records")
    generated_entries: int = Field(
        default=0, description="Number of successfully generated entries"
    )
    failed_entries: int = Field(default=0, description="Number of failed entries")
    templates_used: int = Field(default=0, description="Number of templates used")
    errors: list[str] = Field(
        default_factory=list, description="List of error messages"
    )
    output_paths: dict[str, str] = Field(
        default_factory=dict, description="Paths to generated outputs"
    )
    generation_time: Optional[float] = Field(
        default=None, description="Total generation time in seconds"
    )

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_records == 0:
            return 0.0
        return (self.generated_entries / self.total_records) * 100

    def add_error(self, error_message: str) -> None:
        """Add an error message to the result."""
        self.errors.append(error_message)
        self.failed_entries += 1


class ValidationResult(BaseModel):
    """Result of data validation."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    is_valid: bool = Field(..., description="Whether validation passed")
    errors: list[str] = Field(
        default_factory=list, description="Validation error messages"
    )
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")
    config: Optional["GeneratorConfig"] = Field(
        default=None, description="Validated configuration if successful"
    )

    def add_error(self, message: str) -> None:
        """Add validation error."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add validation warning."""
        self.warnings.append(message)


class TemplateValidationResult(BaseModel):
    """Result of DOCX template validation."""

    model_config = ConfigDict(validate_assignment=True)

    valid: bool = Field(..., description="Whether template is valid")
    error: Optional[str] = Field(None, description="Error message if validation failed")
    placeholders: list[str] = Field(
        default_factory=list, description="Found placeholders in template"
    )
    paragraph_count: int = Field(
        0, ge=0, description="Number of paragraphs in template"
    )
    table_count: int = Field(0, ge=0, description="Number of tables in template")

    @classmethod
    def create_error(cls, error_message: str) -> "TemplateValidationResult":
        """Create a failed validation result with error."""
        return cls(
            valid=False,
            error=error_message,
            placeholders=[],
            paragraph_count=0,
            table_count=0,
        )

    @classmethod
    def create_success(
        cls, placeholders: list[str], paragraph_count: int, table_count: int
    ) -> "TemplateValidationResult":
        """Create a successful validation result."""
        return cls(
            valid=True,
            error=None,
            placeholders=placeholders,
            paragraph_count=paragraph_count,
            table_count=table_count,
        )


# Factory functions for creating models with validation


def create_bbox_from_coords(x1: float, y1: float, x2: float, y2: float) -> BBoxModel:
    """Create a validated BBoxModel from coordinates."""
    return BBoxModel(x1=x1, y1=y1, x2=x2, y2=y2)


def create_xfund_entity(
    entity_id: int, text: str, bbox: BBoxModel, label: str
) -> XFUNDEntity:
    """Create a validated XFUNDEntity."""
    return XFUNDEntity(id=entity_id, text=text, bbox=bbox, label=label)


def validate_config_file(config_path: str) -> ValidationResult:
    """Validate a configuration file."""
    result = ValidationResult(is_valid=True)

    try:
        if not Path(config_path).exists():
            result.add_error(f"Configuration file does not exist: {config_path}")
            return result

        # Load raw JSON data first
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)

        # Try to parse the configuration with validation disabled for paths
        try:
            config = GeneratorConfig.model_construct(**data)
            result.config = config
        except Exception as e:
            result.add_error(f"Error creating configuration: {str(e)}")
            return result

        # Additional validation checks - these are warnings/errors but don't fail config creation
        templates_dir = data.get("templates_dir", "")
        csv_path = data.get("csv_path", "")

        if templates_dir and not Path(templates_dir).exists():
            result.add_error(
                f"Templates directory does not exist: {templates_dir}"
            )

        if csv_path and not Path(csv_path).exists():
            result.add_error(f"CSV file does not exist: {csv_path}")

        fonts_dir = data.get("fonts_dir")
        if fonts_dir and not Path(fonts_dir).exists():
            result.add_warning(f"Fonts directory does not exist: {fonts_dir}")

    except Exception as e:
        result.add_error(f"Error validating configuration: {str(e)}")

    return result


# Example usage and defaults


def get_default_config() -> GeneratorConfig:
    """Get default configuration for development/testing."""
    return GeneratorConfig(
        templates_dir="data/templates_docx",
        csv_path="data/csv/data.csv",
        output_dir="output",
        fonts_dir="fonts/handwritten_fonts",
        document_type=DocumentType.MEDICAL,
        enable_augmentations=True,
        augmentation_difficulty=AugmentationDifficulty.MEDIUM,
        image_dpi=300,
        target_size=1000,
        add_bbox_jitter=True,
        strict_validation=False,
    )


if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    print("Default config created:")
    print(config.model_dump_json(indent=2))

    # Example bbox creation
    bbox = create_bbox_from_coords(10, 20, 100, 80)
    print(f"\nBBox: {bbox.to_list()}")
    print(f"Area: {bbox.area()}")
