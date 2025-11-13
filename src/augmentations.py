"""
Image augmentation utilities with bounding box transformations.
Provides realistic document augmentations while preserving annotation accuracy.
"""

import random
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

from .utils import BBox, apply_bbox_transform

logger = logging.getLogger(__name__)


class DocumentAugmenter:
    """
    Document-specific image augmentations with bbox-aware transformations.
    """
    
    def __init__(
        self,
        target_size: int = 1000,
        enable_noise: bool = True,
        enable_blur: bool = True,
        enable_brightness: bool = True,
        enable_rotation: bool = True,
        enable_perspective: bool = True,
        augmentation_probability: float = 0.7
    ):
        """
        Initialize document augmenter.
        
        Args:
            target_size: Target size for XFUND normalization
            enable_*: Enable/disable specific augmentation types
            augmentation_probability: Probability of applying augmentations
        """
        self.target_size = target_size
        self.augmentation_probability = augmentation_probability
        
        # Create albumentations pipeline
        transforms = []
        
        if enable_noise:
            transforms.extend([
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            ])
        
        if enable_blur:
            transforms.extend([
                A.MotionBlur(blur_limit=3, p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.1),
            ])
        
        if enable_brightness:
            transforms.extend([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.3
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            ])
        
        if enable_rotation:
            transforms.append(
                A.Rotate(limit=3, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=255)
            )
        
        if enable_perspective:
            transforms.append(
                A.Perspective(scale=(0.02, 0.05), p=0.2)
            )
        
        # Add some document-specific augmentations
        transforms.extend([
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.1
            ),
            # Simulate paper texture
            A.Emboss(alpha=(0.1, 0.3), strength=(0.1, 0.3), p=0.1),
        ])
        
        self.transform_pipeline = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc',  # (x_min, y_min, x_max, y_max)
                label_fields=['bbox_labels'],
                min_visibility=0.3  # Minimum visible area to keep bbox
            )
        ) if transforms else None
        
        logger.info(f"Initialized augmenter with {len(transforms)} transform types")
    
    def apply_augmentations(
        self,
        image: np.ndarray,
        annotations: List[Dict[str, Any]],
        apply_probability: Optional[float] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Apply augmentations to image and adjust bounding boxes accordingly.
        
        Args:
            image: Input image as numpy array
            annotations: List of XFUND annotations
            apply_probability: Override default augmentation probability
        
        Returns:
            Tuple of (augmented_image, updated_annotations)
        """
        if apply_probability is None:
            apply_probability = self.augmentation_probability
        
        # Skip augmentation based on probability
        if random.random() > apply_probability:
            return image, annotations
        
        if self.transform_pipeline is None:
            return image, annotations
        
        try:
            # Convert XFUND bboxes to format expected by albumentations
            height, width = image.shape[:2]
            bboxes = []
            bbox_labels = []
            
            for ann in annotations:
                # Denormalize bbox from XFUND format to actual coordinates
                bbox_xfund = BBox(*ann['bbox'])
                bbox_actual = bbox_xfund.denormalize(width, height, self.target_size)
                
                # Convert to pascal_voc format
                bboxes.append([bbox_actual.x1, bbox_actual.y1, bbox_actual.x2, bbox_actual.y2])
                bbox_labels.append(ann['label'])
            
            # Apply transformations
            if bboxes:
                transformed = self.transform_pipeline(
                    image=image,
                    bboxes=bboxes,
                    bbox_labels=bbox_labels
                )
                
                augmented_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_labels = transformed['bbox_labels']
                
                # Update annotations with transformed bboxes
                updated_annotations = []
                for i, (bbox, label) in enumerate(zip(transformed_bboxes, transformed_labels)):
                    # Find original annotation with matching label and similar position
                    original_ann = None
                    for ann in annotations:
                        if ann['label'] == label:
                            original_ann = ann
                            annotations.remove(ann)  # Remove to avoid duplicates
                            break
                    
                    if original_ann:
                        # Convert back to XFUND normalized format
                        aug_height, aug_width = augmented_image.shape[:2]
                        bbox_actual = BBox(bbox[0], bbox[1], bbox[2], bbox[3])
                        bbox_normalized = bbox_actual.normalize(aug_width, aug_height, self.target_size)
                        
                        updated_ann = original_ann.copy()
                        updated_ann['bbox'] = bbox_normalized.to_xfund_format()
                        updated_annotations.append(updated_ann)
                
                # Add any remaining annotations that weren't transformed (edge case)
                updated_annotations.extend(annotations)
                
                logger.debug(f"Applied augmentations: {len(transformed_bboxes)} bboxes transformed")
                return augmented_image, updated_annotations
            
            else:
                # No bboxes to transform, just augment image
                transformed = self.transform_pipeline(image=image, bboxes=[], bbox_labels=[])
                return transformed['image'], annotations
                
        except Exception as e:
            logger.warning(f"Augmentation failed, returning original: {e}")
            return image, annotations
    
    def apply_manual_augmentations(
        self,
        image: np.ndarray,
        annotations: List[Dict[str, Any]],
        augment_config: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Apply specific manual augmentations with custom parameters.
        
        Args:
            image: Input image
            annotations: Annotations list
            augment_config: Configuration for specific augmentations
        
        Returns:
            Augmented image and annotations
        """
        augmented_image = image.copy()
        updated_annotations = annotations.copy()
        
        # Apply scanning artifacts
        if augment_config.get('scanning_artifacts', False):
            augmented_image = self._add_scanning_artifacts(augmented_image)
        
        # Add paper wrinkles/folds
        if augment_config.get('paper_effects', False):
            augmented_image = self._add_paper_effects(augmented_image)
        
        # Simulate ink bleeding
        if augment_config.get('ink_bleeding', False):
            augmented_image = self._add_ink_bleeding(augmented_image)
        
        # Add handwriting variations
        if augment_config.get('handwriting_variation', False):
            # This would require more complex text replacement
            pass
        
        return augmented_image, updated_annotations
    
    def _add_scanning_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Add realistic scanning artifacts."""
        # Add horizontal lines (scanner lines)
        if random.random() < 0.3:
            height, width = image.shape[:2]
            for _ in range(random.randint(1, 3)):
                y = random.randint(0, height - 1)
                intensity = random.randint(200, 240)
                image[y:y+1, :] = np.minimum(image[y:y+1, :], intensity)
        
        # Add slight color cast
        if random.random() < 0.2:
            color_cast = np.random.randint(-5, 5, size=3)
            image = np.clip(image.astype(np.float32) + color_cast, 0, 255).astype(np.uint8)
        
        return image
    
    def _add_paper_effects(self, image: np.ndarray) -> np.ndarray:
        """Add paper texture and fold effects."""
        # Add subtle texture
        if random.random() < 0.4:
            noise = np.random.normal(0, 2, image.shape).astype(np.int8)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add fold/crease effect
        if random.random() < 0.2:
            height, width = image.shape[:2]
            fold_x = random.randint(width // 4, 3 * width // 4)
            
            # Create gradient for fold effect
            x_coords = np.arange(width)
            fold_effect = np.exp(-((x_coords - fold_x) ** 2) / (2 * (width // 20) ** 2))
            fold_effect = (fold_effect * 20).astype(np.uint8)
            
            # Apply fold effect
            if len(image.shape) == 3:
                fold_effect = fold_effect[:, np.newaxis]
            image = np.clip(image.astype(np.int16) - fold_effect, 0, 255).astype(np.uint8)
        
        return image
    
    def _add_ink_bleeding(self, image: np.ndarray) -> np.ndarray:
        """Simulate ink bleeding effects."""
        if random.random() < 0.2:
            # Apply slight blur to dark areas to simulate ink bleeding
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            dark_mask = gray < 200
            
            if dark_mask.any():
                blurred = cv2.GaussianBlur(image, (3, 3), 0.5)
                image[dark_mask] = blurred[dark_mask]
        
        return image


class LightweightAugmenter:
    """
    Lightweight augmenter using PIL for basic transformations.
    Alternative to albumentations for simpler use cases.
    """
    
    def __init__(self, target_size: int = 1000):
        self.target_size = target_size
    
    def apply_basic_augmentations(
        self,
        image_pil: Image.Image,
        annotations: List[Dict[str, Any]]
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """
        Apply basic PIL-based augmentations.
        
        Args:
            image_pil: PIL Image object
            annotations: XFUND annotations
        
        Returns:
            Augmented image and annotations
        """
        augmented = image_pil.copy()
        
        # Brightness adjustment
        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(augmented)
            factor = random.uniform(0.8, 1.2)
            augmented = enhancer.enhance(factor)
        
        # Contrast adjustment
        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(augmented)
            factor = random.uniform(0.8, 1.2)
            augmented = enhancer.enhance(factor)
        
        # Slight blur
        if random.random() < 0.1:
            augmented = augmented.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Add noise
        if random.random() < 0.2:
            augmented = self._add_pil_noise(augmented)
        
        return augmented, annotations
    
    def _add_pil_noise(self, image: Image.Image) -> Image.Image:
        """Add noise using PIL."""
        import numpy as np
        
        # Convert to numpy, add noise, convert back
        img_array = np.array(image)
        noise = np.random.normal(0, 3, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_array)


def create_augmentation_config(
    difficulty: str = 'medium',
    document_type: str = 'medical'
) -> Dict[str, Any]:
    """
    Create augmentation configuration based on difficulty and document type.
    
    Args:
        difficulty: 'light', 'medium', or 'heavy'
        document_type: 'medical', 'legal', 'financial', etc.
    
    Returns:
        Configuration dictionary
    """
    base_config = {
        'enable_noise': True,
        'enable_blur': True,
        'enable_brightness': True,
        'enable_rotation': True,
        'enable_perspective': True,
        'augmentation_probability': 0.7
    }
    
    if difficulty == 'light':
        base_config.update({
            'augmentation_probability': 0.3,
            'enable_perspective': False,
            'enable_rotation': False
        })
    elif difficulty == 'heavy':
        base_config.update({
            'augmentation_probability': 0.9,
            'scanning_artifacts': True,
            'paper_effects': True,
            'ink_bleeding': True
        })
    
    # Document-specific adjustments
    if document_type == 'medical':
        # Medical documents are typically cleaner
        base_config.update({
            'enable_noise': True,
            'ink_bleeding': False
        })
    elif document_type == 'handwritten':
        # Handwritten documents need more variation
        base_config.update({
            'handwriting_variation': True,
            'ink_bleeding': True
        })
    
    return base_config


def validate_augmentation_quality(
    original_annotations: List[Dict[str, Any]],
    augmented_annotations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate that augmentation preserved annotation quality.
    
    Args:
        original_annotations: Original annotations before augmentation
        augmented_annotations: Annotations after augmentation
    
    Returns:
        Validation results
    """
    issues = []
    stats = {
        'annotations_lost': 0,
        'annotations_gained': 0,
        'bbox_shift_stats': []
    }
    
    # Check annotation count changes
    orig_count = len(original_annotations)
    aug_count = len(augmented_annotations)
    
    if aug_count < orig_count:
        stats['annotations_lost'] = orig_count - aug_count
        if stats['annotations_lost'] > orig_count * 0.1:  # Lost more than 10%
            issues.append(f"Lost {stats['annotations_lost']} annotations during augmentation")
    
    elif aug_count > orig_count:
        stats['annotations_gained'] = aug_count - orig_count
        issues.append(f"Gained {stats['annotations_gained']} annotations (unexpected)")
    
    # Check bbox displacement (simplified matching by label)
    for orig_ann in original_annotations:
        label = orig_ann['label']
        text = orig_ann['text']
        
        # Find matching annotation in augmented set
        matching_ann = None
        for aug_ann in augmented_annotations:
            if aug_ann['label'] == label and aug_ann['text'] == text:
                matching_ann = aug_ann
                break
        
        if matching_ann:
            # Calculate bbox center displacement
            orig_bbox = BBox(*orig_ann['bbox'])
            aug_bbox = BBox(*matching_ann['bbox'])
            
            orig_center = ((orig_bbox.x1 + orig_bbox.x2) / 2, (orig_bbox.y1 + orig_bbox.y2) / 2)
            aug_center = ((aug_bbox.x1 + aug_bbox.x2) / 2, (aug_bbox.y1 + aug_bbox.y2) / 2)
            
            displacement = np.sqrt((orig_center[0] - aug_center[0])**2 + 
                                 (orig_center[1] - aug_center[1])**2)
            stats['bbox_shift_stats'].append(displacement)
    
    # Calculate average displacement
    if stats['bbox_shift_stats']:
        avg_displacement = np.mean(stats['bbox_shift_stats'])
        max_displacement = np.max(stats['bbox_shift_stats'])
        
        # Flag large displacements (more than 10% of target size)
        if avg_displacement > 0.1 * 1000:  # 10% of target_size
            issues.append(f"Large average bbox displacement: {avg_displacement:.1f}")
        
        if max_displacement > 0.2 * 1000:  # 20% of target_size
            issues.append(f"Very large max bbox displacement: {max_displacement:.1f}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'stats': stats
    }