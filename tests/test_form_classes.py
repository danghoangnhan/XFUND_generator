"""
Test cases for form classes and OOP inheritance functionality.
"""

import pytest
import json
from unittest.mock import patch

from src.form.base import BaseAnnotation, BaseDataset, Word, LabelType
from src.form.xfund import XFUNDAnnotation, XFUNDDataset
from src.form.funsd import FUNSDAnnotation, FUNSDDataset
from src.form.wildreceipt import WildReceiptAnnotation, WildReceiptDataset


class TestBaseClasses:
    """Test base classes functionality."""

    @pytest.mark.unit
    @pytest.mark.forms
    def test_word_creation(self):
        """Test creating a Word."""
        word = Word(box=[10, 20, 50, 40], text="Test")
        
        assert word.box == [10, 20, 50, 40]
        assert word.text == "Test"

    @pytest.mark.unit
    @pytest.mark.forms
    def test_base_annotation_creation(self):
        """Test creating a BaseAnnotation."""
        annotation = BaseAnnotation(
            box=[10, 20, 100, 80],
            text="Sample text",
            label="question"
        )
        
        assert annotation.box == [10, 20, 100, 80]
        assert annotation.text == "Sample text"
        assert annotation.label == "question"

    @pytest.mark.unit
    @pytest.mark.forms
    def test_label_type_enum(self):
        """Test LabelType enum values."""
        assert LabelType.QUESTION == "question"
        assert LabelType.ANSWER == "answer"
        assert LabelType.OTHER == "other"


class TestXFUNDClasses:
    """Test XFUND-specific classes."""

    @pytest.mark.unit
    @pytest.mark.forms
    def test_xfund_annotation_creation(self, sample_xfund_annotation):
        """Test creating XFUNDAnnotation."""
        assert sample_xfund_annotation.id == 1
        assert sample_xfund_annotation.text == "Patient Name:"
        assert sample_xfund_annotation.label == "question"
        assert sample_xfund_annotation.linking == [[1, 2]]

    @pytest.mark.unit
    @pytest.mark.forms
    def test_xfund_annotation_without_linking(self, sample_word):
        """Test XFUND annotation without linking."""
        annotation = XFUNDAnnotation(
            id=1,
            box=[10, 20, 100, 80],
            text="Answer text",
            label="answer",
            words=[sample_word]
        )
        
        assert annotation.linking == []  # Default empty linking

    @pytest.mark.unit
    @pytest.mark.forms
    def test_xfund_dataset_creation(self, sample_xfund_annotation):
        """Test creating XFUNDDataset."""
        dataset = XFUNDDataset(
            image_path="test.png",
            annotations=[sample_xfund_annotation]
        )
        
        assert dataset.image_path == "test.png"
        assert len(dataset.annotations) == 1
        assert isinstance(dataset.annotations[0], XFUNDAnnotation)

    @pytest.mark.unit
    @pytest.mark.forms
    def test_xfund_dataset_mappings(self):
        """Test XFUND dataset question-answer mappings."""
        word = Word(box=[10, 20, 50, 40], text="Test")
        
        # Create question and answer annotations
        question = XFUNDAnnotation(
            id=1, box=[10, 20, 100, 40], text="Name?", 
            label="question", words=[word], linking=[[1, 2]]
        )
        answer = XFUNDAnnotation(
            id=2, box=[10, 50, 100, 70], text="John",
            label="answer", words=[word]
        )
        
        dataset = XFUNDDataset(
            image_path="test.png",
            annotations=[question, answer]
        )
        
        # Check mappings were built
        assert 1 in dataset.question_to_answer_ids
        assert dataset.question_to_answer_ids[1] == [2]


class TestFUNSDClasses:
    """Test FUNSD-specific classes."""

    @pytest.mark.unit
    @pytest.mark.forms
    def test_funsd_annotation_creation(self, sample_funsd_annotation):
        """Test creating FUNSDAnnotation."""
        assert sample_funsd_annotation.id == 1
        assert sample_funsd_annotation.text == "Name:"
        assert sample_funsd_annotation.label == "question"
        assert sample_funsd_annotation.key_id == 1
        assert sample_funsd_annotation.value_id == 2

    @pytest.mark.unit
    @pytest.mark.forms
    def test_funsd_dataset_creation(self, sample_funsd_annotation):
        """Test creating FUNSDDataset."""
        dataset = FUNSDDataset(
            image_path="test.png",
            annotations=[sample_funsd_annotation]
        )
        
        assert dataset.image_path == "test.png"
        assert len(dataset.annotations) == 1
        assert isinstance(dataset.annotations[0], FUNSDAnnotation)


class TestWildReceiptClasses:
    """Test WildReceipt-specific classes."""

    @pytest.mark.unit
    @pytest.mark.forms
    def test_wildreceipt_annotation_creation(self, sample_wildreceipt_annotation):
        """Test creating WildReceiptAnnotation."""
        assert sample_wildreceipt_annotation.id == 1
        assert sample_wildreceipt_annotation.text == "Total:"
        assert sample_wildreceipt_annotation.label == "question"

    @pytest.mark.unit
    @pytest.mark.forms
    def test_wildreceipt_dataset_creation(self, sample_wildreceipt_annotation):
        """Test creating WildReceiptDataset."""
        dataset = WildReceiptDataset(
            image_path="test.png",
            annotations=[sample_wildreceipt_annotation]
        )
        
        assert dataset.image_path == "test.png"
        assert len(dataset.annotations) == 1
        assert isinstance(dataset.annotations[0], WildReceiptAnnotation)


class TestUnifiedJSONExport:
    """Test the unified to_json() API across all formats."""

    @pytest.mark.unit
    @pytest.mark.forms
    def test_all_formats_have_to_json(self, sample_datasets):
        """Test that all format datasets have to_json() method."""
        for format_name, dataset in sample_datasets.items():
            assert hasattr(dataset, 'to_json')
            assert callable(getattr(dataset, 'to_json'))

    @pytest.mark.unit
    @pytest.mark.forms
    def test_unified_json_export(self, sample_datasets):
        """Test unified JSON export across all formats."""
        results = {}
        
        for format_name, dataset in sample_datasets.items():
            json_output = dataset.to_json()
            
            # Should be valid JSON
            parsed = json.loads(json_output)
            assert isinstance(parsed, dict)
            assert "annotations" in parsed
            
            # Store for comparison
            results[format_name] = parsed
        
        # All should have annotations
        for format_name, result in results.items():
            assert len(result["annotations"]) == 1

    @pytest.mark.unit
    @pytest.mark.forms
    def test_xfund_specific_fields(self, sample_datasets):
        """Test that XFUND format includes linking field."""
        xfund_json = sample_datasets['xfund'].to_json()
        parsed = json.loads(xfund_json)
        annotation = parsed["annotations"][0]
        
        assert "linking" in annotation
        assert annotation["linking"] == [[1, 2]]

    @pytest.mark.unit
    @pytest.mark.forms
    def test_funsd_specific_fields(self, sample_datasets):
        """Test that FUNSD format includes key/value IDs."""
        funsd_json = sample_datasets['funsd'].to_json()
        parsed = json.loads(funsd_json)
        annotation = parsed["annotations"][0]
        
        assert "key_id" in annotation
        assert "value_id" in annotation
        assert annotation["key_id"] == 1
        assert annotation["value_id"] == 2

    @pytest.mark.unit
    @pytest.mark.forms
    def test_wildreceipt_minimal_fields(self, sample_datasets):
        """Test that WildReceipt format has minimal fields."""
        wild_json = sample_datasets['wildreceipt'].to_json()
        parsed = json.loads(wild_json)
        annotation = parsed["annotations"][0]
        
        # Should NOT have format-specific fields
        assert "linking" not in annotation
        assert "key_id" not in annotation
        assert "value_id" not in annotation


class TestPolymorphism:
    """Test polymorphic behavior of form classes."""

    @pytest.mark.unit
    @pytest.mark.forms
    def test_polymorphic_to_json(self, sample_datasets):
        """Test that to_json() works polymorphically."""
        def export_any_dataset(dataset):
            """Function that works with any dataset type."""
            return dataset.to_json()
        
        # Should work with all format types
        for format_name, dataset in sample_datasets.items():
            json_output = export_any_dataset(dataset)
            assert isinstance(json_output, str)
            assert len(json_output) > 0

    @pytest.mark.unit
    @pytest.mark.forms
    def test_isinstance_base_dataset(self, sample_datasets):
        """Test that all datasets are instances of BaseDataset."""
        for format_name, dataset in sample_datasets.items():
            assert isinstance(dataset, BaseDataset)

    @pytest.mark.unit
    @pytest.mark.forms
    def test_format_specific_behavior(self, sample_datasets):
        """Test that each format maintains its specific behavior."""
        json_outputs = {}
        
        for format_name, dataset in sample_datasets.items():
            json_outputs[format_name] = json.loads(dataset.to_json())
        
        # XFUND should have longest output (with linking)
        xfund_len = len(json.dumps(json_outputs['xfund']))
        funsd_len = len(json.dumps(json_outputs['funsd']))
        wild_len = len(json.dumps(json_outputs['wildreceipt']))
        
        # XFUND typically has more fields than others
        assert xfund_len >= funsd_len
        assert funsd_len >= wild_len


class TestInheritanceBenefits:
    """Test the benefits of OOP inheritance architecture."""

    @pytest.mark.unit
    @pytest.mark.forms
    def test_no_redundant_methods(self, sample_datasets):
        """Test that there are no format-specific JSON methods."""
        for format_name, dataset in sample_datasets.items():
            # Should NOT have format-specific methods
            assert not hasattr(dataset, 'to_xfund_json')
            assert not hasattr(dataset, 'to_funsd_json') 
            assert not hasattr(dataset, 'to_wildreceipt_json')
            
            # Should ONLY have unified method
            assert hasattr(dataset, 'to_json')

    @pytest.mark.unit
    @pytest.mark.forms
    def test_template_method_pattern(self, sample_datasets):
        """Test that Template Method pattern is implemented."""
        for format_name, dataset in sample_datasets.items():
            # Should have private formatting method
            assert hasattr(dataset, '_format_annotation_for_export')
            
            # Base method should exist
            assert hasattr(dataset, 'to_json')

    @pytest.mark.integration
    @pytest.mark.forms
    def test_extensibility(self, sample_word):
        """Test that new formats can be easily added."""
        # Simulate adding a new format
        class CustomAnnotation(BaseAnnotation):
            custom_field: str = "default"
        
        class CustomDataset(BaseDataset):
            annotations: list[CustomAnnotation]
            
            def _format_annotation_for_export(self, annotation) -> dict:
                base_format = super()._format_annotation_for_export(annotation)
                base_format["custom_field"] = annotation.custom_field
                return base_format
        
        # Test that it works with the unified API
        custom_annotation = CustomAnnotation(
            box=[10, 20, 100, 80],
            text="Custom text",
            label="custom",
            custom_field="custom_value"
        )
        
        custom_dataset = CustomDataset(
            image_path="test.png",
            annotations=[custom_annotation]
        )
        
        # Should work with unified API
        json_output = custom_dataset.to_json()
        parsed = json.loads(json_output)
        
        assert "annotations" in parsed
        assert len(parsed["annotations"]) == 1
        assert parsed["annotations"][0]["custom_field"] == "custom_value"