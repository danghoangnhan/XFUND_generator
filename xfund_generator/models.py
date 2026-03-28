"""
Pydantic models for XFUND dataset generation.
Provides data validation, serialization, and type safety.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


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
        return [round(self.x1), round(self.y1), round(self.x2), round(self.y2)]

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

    def denormalize(
        self, img_width: int, img_height: int, source_size: int = 1000
    ) -> "BBoxModel":
        """Convert from normalized (0-source_size) scale to image coordinates."""
        return BBoxModel(
            x1=(self.x1 / source_size) * img_width,
            y1=(self.y1 / source_size) * img_height,
            x2=(self.x2 / source_size) * img_width,
            y2=(self.y2 / source_size) * img_height,
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
    """Configuration for document augmentations.

    This model defines all augmentation settings used by DocumentAugmenter.
    Supports loading from YAML files and creating presets by difficulty level.
    """

    model_config = ConfigDict(validate_assignment=True)

    # Metadata
    difficulty: AugmentationDifficulty = Field(
        default=AugmentationDifficulty.MEDIUM,
        description="Augmentation difficulty preset",
    )
    document_type: DocumentType = Field(
        default=DocumentType.MEDICAL,
        description="Target document type for augmentation tuning",
    )

    # Core augmentation toggles (used by DocumentAugmenter)
    enable_noise: bool = Field(default=True, description="Enable noise augmentations")
    enable_blur: bool = Field(default=True, description="Enable blur augmentations")
    enable_brightness: bool = Field(
        default=True, description="Enable brightness/contrast augmentations"
    )
    enable_rotation: bool = Field(
        default=True, description="Enable rotation augmentations"
    )
    enable_perspective: bool = Field(
        default=True, description="Enable perspective transform"
    )
    augmentation_probability: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Overall probability of applying augmentations",
    )

    # Manual augmentation options (used by apply_manual_augmentations)
    scanning_artifacts: bool = Field(
        default=False, description="Add scanning artifact effects"
    )
    paper_effects: bool = Field(
        default=False, description="Add paper texture and fold effects"
    )
    ink_bleeding: bool = Field(default=False, description="Simulate ink bleeding")
    handwriting_variation: bool = Field(
        default=False, description="Enable handwriting style variations"
    )

    # Target size for XFUND normalization
    target_size: int = Field(
        default=1000, ge=224, description="Target size for bbox normalization"
    )

    # Bbox tracking during augmentation
    min_visibility: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum visible area ratio to keep a bbox after augmentation",
    )

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> "AugmentationConfig":
        """Load augmentation config from a YAML file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            AugmentationConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        return cls(**data)

    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """Save augmentation config to a YAML file.

        Args:
            file_path: Path to save YAML configuration
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_difficulty(
        cls,
        difficulty: Union[str, AugmentationDifficulty] = "medium",
        document_type: Union[str, DocumentType] = "medical",
    ) -> "AugmentationConfig":
        """Create config from difficulty preset and document type.

        Loads preset from config/augmentation_presets/{difficulty}.yaml.

        Args:
            difficulty: 'easy', 'medium', 'hard', or 'extreme'
            document_type: 'medical', 'form', 'invoice', etc.

        Returns:
            AugmentationConfig with appropriate settings
        """
        from .utils import get_config_dir

        # Normalize inputs
        if isinstance(difficulty, AugmentationDifficulty):
            difficulty_str = difficulty.value
        else:
            difficulty_str = difficulty.lower()

        if isinstance(document_type, str):
            document_type = DocumentType(document_type.lower())

        # Try loading preset from YAML
        preset_path = (
            get_config_dir() / f"augmentation_presets/{difficulty_str}.yaml"
        )
        if preset_path.exists():
            config = cls.from_yaml(preset_path)
            config.document_type = document_type
            config.difficulty = AugmentationDifficulty(difficulty_str)
            return config

        # Fallback to defaults if preset file not found
        return cls(
            difficulty=AugmentationDifficulty(difficulty_str),
            document_type=document_type,
        )


class LayoutField(BaseModel):
    """Single field position in a layout template."""

    x1: float
    y1: float
    x2: float
    y2: float

    def to_bbox(self) -> "BBoxModel":
        """Convert to BBoxModel."""
        return BBoxModel(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)


class LayoutConfig(BaseModel):
    """Layout configuration mapping field names to bounding boxes."""

    fields: dict[str, LayoutField]

    @classmethod
    def from_json_file(cls, path: str) -> "LayoutConfig":
        """Load layout from a JSON file mapping field names to [x1,y1,x2,y2]."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Layout JSON not found: {path}")
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        fields = {}
        for name, coords in raw.items():
            if not isinstance(coords, list) or len(coords) != 4:
                raise ValueError(
                    f"Invalid bbox format for field '{name}': {coords}"
                )
            fields[name] = LayoutField(
                x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]
            )
        return cls(fields=fields)

    def get_field_bbox(self, name: str) -> Optional["BBoxModel"]:
        """Get a field's bounding box by name, or None if not found."""
        field = self.fields.get(name)
        return field.to_bbox() if field else None

    def field_names(self) -> list[str]:
        """Return list of field names in this layout."""
        return list(self.fields.keys())

    def to_raw_dict(self) -> dict[str, list[float]]:
        """Convert back to raw dict format for backward compatibility."""
        return {
            name: [f.x1, f.y1, f.x2, f.y2]
            for name, f in self.fields.items()
        }


class EntryResult(BaseModel):
    """Result of generating a single dataset entry."""

    success: bool
    entry_id: str = ""
    image_path: str = ""
    annotation_path: str = ""
    annotation_count: int = 0
    error: Optional[str] = None

    @classmethod
    def ok(
        cls,
        entry_id: str,
        image_path: str,
        annotation_path: str,
        annotation_count: int = 0,
    ) -> "EntryResult":
        return cls(
            success=True,
            entry_id=entry_id,
            image_path=image_path,
            annotation_path=annotation_path,
            annotation_count=annotation_count,
        )

    @classmethod
    def fail(cls, error: str, entry_id: str = "") -> "EntryResult":
        return cls(success=False, error=error, entry_id=entry_id)


class SetupValidationResult(BaseModel):
    """Result of validating generator setup."""

    valid: bool
    issues: list[str] = Field(default_factory=list)
    templates_found: int = 0


class AugmentationQualityStats(BaseModel):
    """Statistics from augmentation quality validation."""

    annotations_lost: int = 0
    annotations_gained: int = 0
    bbox_shift_stats: list[float] = Field(default_factory=list)


class AugmentationQualityResult(BaseModel):
    """Result of augmentation quality validation."""

    valid: bool
    issues: list[str] = Field(default_factory=list)
    stats: AugmentationQualityStats = Field(
        default_factory=AugmentationQualityStats
    )


class BatchEntryResult(BaseModel):
    """Result of a single entry in batch generation."""

    template_name: str
    status: str  # "success" or "failed"
    image_path: str = ""
    annotation_path: str = ""
    annotations_count: int = 0
    qa_pairs: int = 0
    error: Optional[str] = None


class TemplateInfo(BaseModel):
    """Information about a discovered template pair (DOCX + layout JSON)."""

    model_config = ConfigDict(validate_assignment=True)

    name: str = Field(..., description="Template name (without extension)")
    docx_path: str = Field(..., description="Path to DOCX template file")
    layout_path: str = Field(..., description="Path to layout JSON file")


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
    department_text: Optional[str] = Field(
        default=None, description="Medical department"
    )
    diagnose_text: Optional[str] = Field(default=None, description="Diagnosis")
    doctor_comment_text: Optional[str] = Field(
        default=None, description="Doctor comments"
    )

    # Template-related fields for XFUND form integration
    template_name: Optional[str] = Field(default=None, description="Template name")
    field_name: Optional[str] = Field(default=None, description="Field name")
    field_value: Optional[str] = Field(default=None, description="Field value")
    bbox: Optional[str] = Field(default=None, description="Bounding box coordinates")

    # Additional fields for other document types
    additional_fields: Optional[dict[str, str]] = Field(
        default_factory=dict, description="Additional custom fields"
    )

    def get_field(self, field_name: str) -> str:
        """Get field value by name with fallback to additional_fields."""
        if hasattr(self, field_name):
            return getattr(self, field_name) or ""
        if self.additional_fields is not None:
            return self.additional_fields.get(field_name, "")
        return ""

    def get_bbox_coordinates(self) -> Optional[BBoxModel]:
        """Parse bbox string and return as BBoxModel, or None if invalid."""
        if self.bbox is None:
            return None
        try:
            parts = [float(x.strip()) for x in self.bbox.split(",")]
            if len(parts) == 4:
                return BBoxModel(x1=parts[0], y1=parts[1], x2=parts[2], y2=parts[3])
        except (ValueError, AttributeError):
            pass
        return None


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

    # Linking settings (for Q/A annotation pairing)
    max_linking_distance: int = Field(
        default=100,
        ge=1,
        description="Maximum pixel distance for Q/A linking",
    )
    max_linked_answers: int = Field(
        default=3,
        ge=1,
        description="Maximum number of answers to link per question",
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


class WordAnnotation(BaseModel):
    """Pydantic model for word-level XFUND annotation."""

    model_config = ConfigDict(validate_assignment=True)

    text: str = Field(..., description="Word text content")
    bbox: list[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    label: str = Field(..., description="Field label/category")

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v

    @field_validator("bbox")
    @classmethod
    def validate_bbox_format(cls, v: list[int]) -> list[int]:
        if len(v) != 4:
            raise ValueError("Bbox must have exactly 4 coordinates [x1, y1, x2, y2]")
        return v

    @model_validator(mode="after")
    def validate_bbox_coordinates(self) -> "WordAnnotation":
        x1, y1, x2, y2 = self.bbox
        if x1 >= x2:
            raise ValueError(f"x1 ({x1}) must be less than x2 ({x2})")
        if y1 >= y2:
            raise ValueError(f"y1 ({y1}) must be less than y2 ({y2})")
        return self

    def validate_bounds(self, max_value: int = 1000) -> list[str]:
        """Validate bbox coordinates are within bounds. Returns list of issues."""
        issues = []
        for i, coord in enumerate(self.bbox):
            if coord < 0 or coord > max_value:
                issues.append(
                    f"Coordinate {i} ({coord}) out of bounds [0, {max_value}]"
                )
        return issues

    def intersects(self, other: "WordAnnotation") -> bool:
        """Check if this annotation's bbox overlaps with another."""
        x1_a, y1_a, x2_a, y2_a = self.bbox
        x1_b, y1_b, x2_b, y2_b = other.bbox

        # Return True if bboxes overlap
        return not (x2_a <= x1_b or x2_b <= x1_a or y2_a <= y1_b or y2_b <= y1_a)



class XFUNDEntry(BaseModel):
    """Pydantic model for a complete XFUND dataset entry.

    Represents a single document with its image and word-level annotations.
    Used by WordRenderer.create_xfund_entry for validated entry creation.
    """

    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(..., description="Unique entry identifier")
    image: str = Field(..., description="Relative path to image file")
    annotations: list[WordAnnotation] = Field(
        default_factory=list, description="List of word-level annotations"
    )

    @field_validator("id")
    @classmethod
    def validate_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Entry ID cannot be empty")
        return v.strip()

    @field_validator("image")
    @classmethod
    def validate_image_path(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Image path cannot be empty")
        return v


    @property
    def annotation_count(self) -> int:
        """Return the number of annotations."""
        return len(self.annotations)

    @property
    def unique_labels(self) -> set[str]:
        """Return set of unique labels in this entry."""
        return {ann.label for ann in self.annotations}


class AnnotationStats(BaseModel):
    """Statistics about a collection of annotations."""

    model_config = ConfigDict(validate_assignment=True)

    total_annotations: int = Field(default=0, description="Total number of annotations")
    unique_labels: int = Field(default=0, description="Number of unique labels")
    words_per_label: dict[str, int] = Field(
        default_factory=dict, description="Word count per label"
    )
    bbox_overlaps: list[tuple[int, int]] = Field(
        default_factory=list, description="Pairs of overlapping annotation indices"
    )


class AnnotationValidationResult(BaseModel):
    """Result of annotation validation."""

    model_config = ConfigDict(validate_assignment=True)

    valid: bool = Field(..., description="Whether all annotations are valid")
    issues: list[str] = Field(
        default_factory=list, description="Validation error messages"
    )
    stats: AnnotationStats = Field(
        default_factory=AnnotationStats, description="Annotation statistics"
    )

    @classmethod
    def create_valid(cls, stats: AnnotationStats) -> "AnnotationValidationResult":
        """Create a successful validation result."""
        return cls(valid=True, issues=[], stats=stats)

    @classmethod
    def create_invalid(
        cls, issues: list[str], stats: AnnotationStats
    ) -> "AnnotationValidationResult":
        """Create a failed validation result."""
        return cls(valid=len(issues) == 0, issues=issues, stats=stats)


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
            result.add_error(f"Templates directory does not exist: {templates_dir}")

        if csv_path and not Path(csv_path).exists():
            result.add_error(f"CSV file does not exist: {csv_path}")

        fonts_dir = data.get("fonts_dir")
        if fonts_dir and not Path(fonts_dir).exists():
            result.add_warning(f"Fonts directory does not exist: {fonts_dir}")

    except json.JSONDecodeError as e:
        result.add_error(f"Invalid JSON in configuration file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error validating configuration: {e}", exc_info=True)
        result.add_error(f"Error validating configuration: {type(e).__name__}: {e}")

    return result


def validate_annotations(
    annotations: list[dict[str, Any]],
    target_size: int = 1000,
    check_overlaps: bool = True,
) -> AnnotationValidationResult:
    """
    Validate a list of word annotations using Pydantic models.

    Args:
        annotations: List of annotation dictionaries with text, bbox, label
        target_size: Maximum allowed bbox coordinate value
        check_overlaps: Whether to check for overlapping bboxes

    Returns:
        AnnotationValidationResult with validation status and statistics
    """
    issues: list[str] = []
    valid_annotations: list[WordAnnotation] = []
    label_counts: dict[str, int] = {}

    # Validate each annotation
    for i, ann in enumerate(annotations):
        try:
            word_ann = WordAnnotation.model_validate(ann)
            valid_annotations.append(word_ann)

            # Count labels
            label_counts[word_ann.label] = label_counts.get(word_ann.label, 0) + 1

            # Check bounds
            bounds_issues = word_ann.validate_bounds(target_size)
            for issue in bounds_issues:
                issues.append(f"Annotation {i}: {issue}")

        except (ValueError, KeyError) as e:
            issues.append(f"Annotation {i}: {e}")

    # Check for overlapping bboxes
    overlaps: list[tuple[int, int]] = []
    if check_overlaps:
        for i, ann1 in enumerate(valid_annotations):
            for j, ann2 in enumerate(valid_annotations[i + 1 :], i + 1):
                if ann1.intersects(ann2):
                    overlaps.append((i, j))

    stats = AnnotationStats(
        total_annotations=len(annotations),
        unique_labels=len(label_counts),
        words_per_label=label_counts,
        bbox_overlaps=overlaps,
    )

    return AnnotationValidationResult(
        valid=len(issues) == 0,
        issues=issues,
        stats=stats,
    )


# Example usage and defaults


def get_default_config() -> GeneratorConfig:
    """Get default configuration for development/testing.

    Loads defaults from config/defaults.yaml.
    """
    from .utils import load_yaml_config

    data = load_yaml_config("defaults.yaml")
    return GeneratorConfig(**data)


if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    print("Default config created:")
    print(config.model_dump_json(indent=2))

    # Example bbox creation
    bbox = BBoxModel(x1=10, y1=20, x2=100, y2=80)
    print(f"\nBBox: {bbox.to_list()}")
    print(f"Area: {bbox.area()}")
