"""
Test cases for BBoxModel improvements (denormalize, rounding).
"""

import pytest

from xfund_generator.models import BBoxModel


class TestBBoxModelRounding:
    """Test that to_xfund_format uses round() not int()."""

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_xfund_format_rounds_up(self):
        """Values >= 0.5 should round up."""
        bbox = BBoxModel(x1=10.7, y1=20.5, x2=100.9, y2=80.6)
        result = bbox.to_xfund_format()
        assert result == [11, 20, 101, 81]

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_xfund_format_rounds_down(self):
        """Values < 0.5 should round down."""
        bbox = BBoxModel(x1=10.2, y1=20.4, x2=100.1, y2=80.3)
        result = bbox.to_xfund_format()
        assert result == [10, 20, 100, 80]

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_xfund_format_integers_unchanged(self):
        """Integer-valued floats should remain the same."""
        bbox = BBoxModel(x1=10.0, y1=20.0, x2=100.0, y2=80.0)
        result = bbox.to_xfund_format()
        assert result == [10, 20, 100, 80]


class TestBBoxModelDenormalize:
    """Test BBoxModel.denormalize() method."""

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_denormalize_basic(self):
        """Test basic denormalization from XFUND scale to image coordinates."""
        bbox = BBoxModel(x1=500.0, y1=250.0, x2=750.0, y2=500.0)
        result = bbox.denormalize(img_width=2000, img_height=3000, source_size=1000)

        assert result.x1 == pytest.approx(1000.0)
        assert result.y1 == pytest.approx(750.0)
        assert result.x2 == pytest.approx(1500.0)
        assert result.y2 == pytest.approx(1500.0)

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_normalize_denormalize_roundtrip(self):
        """Normalizing then denormalizing should return original coordinates."""
        original = BBoxModel(x1=100.0, y1=200.0, x2=300.0, y2=400.0)

        normalized = original.normalize(img_width=1000, img_height=2000)
        restored = normalized.denormalize(img_width=1000, img_height=2000)

        assert restored.x1 == pytest.approx(original.x1)
        assert restored.y1 == pytest.approx(original.y1)
        assert restored.x2 == pytest.approx(original.x2)
        assert restored.y2 == pytest.approx(original.y2)

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_denormalize_preserves_proportions(self):
        """Denormalized bbox should have correct proportions."""
        bbox = BBoxModel(x1=0.0, y1=0.0, x2=500.0, y2=500.0)
        result = bbox.denormalize(img_width=2000, img_height=1000, source_size=1000)

        # x should span half the image width
        assert result.x2 == pytest.approx(1000.0)
        # y should span half the image height
        assert result.y2 == pytest.approx(500.0)


class TestBBoxModelConfigurability:
    """Test new configurable fields."""

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_generator_config_linking_defaults(self):
        """Test default values for linking config."""
        from xfund_generator.models import GeneratorConfig

        config = GeneratorConfig(
            templates_dir="data/templates_docx",
            csv_path="data/csv/data.csv",
            output_dir="output",
        )
        assert config.max_linking_distance == 100
        assert config.max_linked_answers == 3

    @pytest.mark.unit
    @pytest.mark.pydantic
    def test_generator_config_custom_linking(self):
        """Test custom linking config values."""
        from xfund_generator.models import GeneratorConfig

        config = GeneratorConfig(
            templates_dir="data/templates_docx",
            csv_path="data/csv/data.csv",
            output_dir="output",
            max_linking_distance=200,
            max_linked_answers=5,
        )
        assert config.max_linking_distance == 200
        assert config.max_linked_answers == 5
