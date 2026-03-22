"""
Test cases for augmentation pipeline and bbox tracking.
"""

import numpy as np
import pytest

from xfund_generator.augmentations import (
    DocumentAugmenter,
    validate_augmentation_quality,
)
from xfund_generator.models import AugmentationConfig, WordAnnotation


@pytest.fixture
def sample_image():
    """Create a simple test image (white background with some text-like marks)."""
    image = np.ones((500, 500, 3), dtype=np.uint8) * 255
    # Add some dark rectangles to simulate text
    image[50:80, 50:200] = 30  # Simulate text region 1
    image[100:130, 50:250] = 30  # Simulate text region 2
    image[150:180, 50:180] = 30  # Simulate text region 3
    return image


@pytest.fixture
def sample_annotations():
    """Create sample WordAnnotation list with multiple same-label fields."""
    return [
        WordAnnotation(text="Name:", bbox=[100, 100, 400, 160], label="QUESTION"),
        WordAnnotation(text="John Doe", bbox=[100, 200, 500, 260], label="ANSWER"),
        WordAnnotation(text="Age:", bbox=[100, 300, 400, 360], label="QUESTION"),
        WordAnnotation(text="30", bbox=[100, 400, 500, 460], label="ANSWER"),
    ]


@pytest.fixture
def augmenter_no_transform():
    """Create an augmenter with minimal transforms for predictable testing."""
    config = AugmentationConfig(
        enable_noise=False,
        enable_blur=False,
        enable_brightness=False,
        enable_rotation=False,
        enable_perspective=False,
        augmentation_probability=1.0,
    )
    return DocumentAugmenter(config=config)


class TestAnnotationMatching:
    """Test that augmentation correctly tracks annotations by index."""

    @pytest.mark.unit
    def test_annotations_preserve_text_after_augmentation(
        self, sample_image, sample_annotations
    ):
        """Critical regression test: annotations must preserve their text
        even when multiple annotations share the same label."""
        config = AugmentationConfig(
            enable_noise=True,
            enable_blur=False,
            enable_brightness=False,
            enable_rotation=False,
            enable_perspective=False,
            augmentation_probability=1.0,
        )
        augmenter = DocumentAugmenter(config=config)

        _, updated_annotations = augmenter.apply_augmentations(
            sample_image, sample_annotations
        )

        # Extract texts - order and text content must be preserved
        original_texts = [ann.text for ann in sample_annotations]
        updated_texts = [ann.text for ann in updated_annotations]

        assert original_texts == updated_texts, (
            f"Annotation texts were scrambled after augmentation. "
            f"Original: {original_texts}, Got: {updated_texts}"
        )

    @pytest.mark.unit
    def test_same_label_annotations_not_swapped(
        self, sample_image, sample_annotations
    ):
        """Verify that two ANSWER annotations don't get each other's bbox."""
        config = AugmentationConfig(
            enable_noise=True,
            enable_blur=False,
            enable_brightness=False,
            enable_rotation=False,
            enable_perspective=False,
            augmentation_probability=1.0,
        )
        augmenter = DocumentAugmenter(config=config)

        _, updated_annotations = augmenter.apply_augmentations(
            sample_image, sample_annotations
        )

        # Find the two ANSWER annotations
        answers = [a for a in updated_annotations if a.label == "ANSWER"]
        assert len(answers) == 2

        # First answer should still be "John Doe", second "30"
        assert answers[0].text == "John Doe"
        assert answers[1].text == "30"

    @pytest.mark.unit
    def test_labels_preserved_after_augmentation(
        self, sample_image, sample_annotations
    ):
        """Labels must be preserved through augmentation."""
        config = AugmentationConfig(
            enable_noise=True,
            enable_blur=False,
            enable_brightness=False,
            enable_rotation=False,
            enable_perspective=False,
            augmentation_probability=1.0,
        )
        augmenter = DocumentAugmenter(config=config)

        _, updated_annotations = augmenter.apply_augmentations(
            sample_image, sample_annotations
        )

        original_labels = [ann.label for ann in sample_annotations]
        updated_labels = [ann.label for ann in updated_annotations]

        assert original_labels == updated_labels

    @pytest.mark.unit
    def test_augmentation_with_empty_annotations(self, sample_image):
        """Augmentation should work with empty annotation list."""
        config = AugmentationConfig(
            enable_noise=True,
            enable_blur=False,
            enable_brightness=False,
            enable_rotation=False,
            enable_perspective=False,
            augmentation_probability=1.0,
        )
        augmenter = DocumentAugmenter(config=config)

        aug_image, annotations = augmenter.apply_augmentations(sample_image, [])
        assert annotations == []
        assert aug_image is not None

    @pytest.mark.unit
    def test_augmentation_probability_zero(self, sample_image, sample_annotations):
        """With probability 0, image and annotations should be unchanged."""
        config = AugmentationConfig(
            enable_noise=True,
            augmentation_probability=0.0,
        )
        augmenter = DocumentAugmenter(config=config)

        aug_image, updated_annotations = augmenter.apply_augmentations(
            sample_image, sample_annotations, apply_probability=0.0
        )

        # Should return originals
        np.testing.assert_array_equal(aug_image, sample_image)
        assert updated_annotations == sample_annotations


class TestAugmentationConfig:
    """Test AugmentationConfig integration with augmenter."""

    @pytest.mark.unit
    def test_min_visibility_from_config(self):
        """min_visibility should be read from config."""
        config = AugmentationConfig(
            min_visibility=0.5,
            enable_noise=True,
            enable_blur=False,
            enable_brightness=False,
            enable_rotation=False,
            enable_perspective=False,
        )
        augmenter = DocumentAugmenter(config=config)

        # The transform pipeline should use config's min_visibility
        assert augmenter.config.min_visibility == 0.5

    @pytest.mark.unit
    def test_default_min_visibility(self):
        """Default min_visibility should be 0.3."""
        config = AugmentationConfig()
        assert config.min_visibility == 0.3


class TestAugmentationQualityValidation:
    """Test augmentation quality validation."""

    @pytest.mark.unit
    def test_validate_with_matching_counts(self):
        """Validation should pass when annotation counts match."""
        original = [
            WordAnnotation(text="Test", bbox=[10, 10, 100, 50], label="ANSWER"),
        ]
        augmented = [
            WordAnnotation(text="Test", bbox=[12, 11, 98, 49], label="ANSWER"),
        ]

        result = validate_augmentation_quality(original, augmented)
        assert result.valid is True
        assert result.stats.annotations_lost == 0
        assert result.stats.annotations_gained == 0

    @pytest.mark.unit
    def test_validate_with_mismatched_counts(self):
        """Validation should flag when many annotations are lost."""
        original = [
            WordAnnotation(text="Test1", bbox=[10, 10, 100, 50], label="ANSWER"),
            WordAnnotation(text="Test2", bbox=[10, 60, 100, 100], label="ANSWER"),
        ]
        augmented = [
            WordAnnotation(text="Test1", bbox=[12, 11, 98, 49], label="ANSWER"),
        ]

        result = validate_augmentation_quality(original, augmented)
        assert result.stats.annotations_lost == 1
