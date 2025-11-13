# Advanced Examples

This document provides comprehensive examples for advanced use cases of the XFUND Generator.

## Advanced Configuration

### Multi-Template Configuration

Handle multiple document templates with different layouts:

```python
from src.models import GeneratorConfig
from src.generate_dataset import XFUNDGenerator
import json

# Advanced configuration with multiple templates
config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/multi_template_data.csv",
    output_dir="output/advanced",
    image_dpi=600,  # High quality
    enable_augmentations=True,
    max_workers=8,
    batch_size=15,
    strict_validation=True,
    generate_debug_overlays=True
)

# Create generator with advanced settings
generator = XFUNDGenerator(config)

# Pre-validate all templates
templates = generator.get_templates()
print(f"Found templates: {templates}")

for template in templates:
    print(f"Validating template: {template}")
    # Custom validation logic here

# Generate with monitoring
result = generator.generate_dataset()
print(f"Advanced generation complete: {result.generated_entries} entries")
```

### Dynamic Configuration Loading

Load configurations based on environment or parameters:

```python
import os
from src.models import GeneratorConfig, get_default_config

def load_config_for_environment(env: str = None) -> GeneratorConfig:
    """Load configuration based on environment."""
    if env is None:
        env = os.getenv('XFUND_ENV', 'development')
    
    config_files = {
        'development': 'config_dev.json',
        'staging': 'config_staging.json', 
        'production': 'config_prod.json'
    }
    
    config_file = config_files.get(env)
    if config_file and os.path.exists(config_file):
        return GeneratorConfig.from_json_file(config_file)
    
    # Fallback to default with environment-specific overrides
    config = get_default_config()
    
    if env == 'production':
        config.image_dpi = 600
        config.strict_validation = True
        config.enable_augmentations = False
    elif env == 'development':
        config.generate_debug_overlays = True
        config.max_workers = 2
    
    return config

# Usage
config = load_config_for_environment('production')
generator = XFUNDGenerator(config)
```

### Custom Validation Rules

Implement custom validation logic:

```python
from src.models import DataRecord, ValidationResult
from typing import List
import re

class CustomValidator:
    """Custom validation rules for specific use cases."""
    
    @staticmethod
    def validate_medical_data(records: List[DataRecord]) -> ValidationResult:
        """Validate medical form data."""
        errors = []
        warnings = []
        
        for i, record in enumerate(records):
            # Validate patient ID format
            if record.field_name == 'patient_id':
                if not re.match(r'^P\d{6}$', record.field_value):
                    errors.append(f"Row {i}: Invalid patient ID format: {record.field_value}")
            
            # Validate date fields
            elif 'date' in record.field_name.lower():
                if not re.match(r'^\d{2}/\d{2}/\d{4}$', record.field_value):
                    warnings.append(f"Row {i}: Date format should be MM/DD/YYYY: {record.field_value}")
            
            # Validate phone numbers
            elif record.field_name == 'phone':
                if not re.match(r'^\(\d{3}\) \d{3}-\d{4}$', record.field_value):
                    errors.append(f"Row {i}: Invalid phone format: {record.field_value}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def validate_bbox_placement(records: List[DataRecord]) -> ValidationResult:
        """Validate bounding box placement and overlap."""
        errors = []
        
        # Group by template
        template_groups = {}
        for record in records:
            if record.template_name not in template_groups:
                template_groups[record.template_name] = []
            template_groups[record.template_name].append(record)
        
        for template_name, template_records in template_groups.items():
            # Check for overlapping bounding boxes
            for i, record1 in enumerate(template_records):
                bbox1 = record1.get_bbox_coordinates()
                
                for j, record2 in enumerate(template_records[i+1:], i+1):
                    bbox2 = record2.get_bbox_coordinates()
                    
                    # Check for overlap
                    if bbox_overlap(bbox1, bbox2):
                        errors.append(
                            f"Template {template_name}: Overlapping bboxes for "
                            f"'{record1.field_name}' and '{record2.field_name}'"
                        )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=[]
        )

def bbox_overlap(bbox1, bbox2):
    """Check if two bounding boxes overlap."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

# Usage example
from src.utils import load_csv_data_as_models

records = load_csv_data_as_models("data/csv/medical_data.csv")

# Apply custom validations
medical_validation = CustomValidator.validate_medical_data(records)
bbox_validation = CustomValidator.validate_bbox_placement(records)

if not medical_validation.is_valid:
    print("Medical validation errors:")
    for error in medical_validation.errors:
        print(f"  - {error}")

if bbox_validation.warnings:
    print("Placement warnings:")
    for warning in bbox_validation.warnings:
        print(f"  - {warning}")
```

## Batch Processing

### Process Multiple Datasets

Generate multiple datasets with different configurations:

```python
from src.models import GeneratorConfig
from src.generate_dataset import XFUNDGenerator
import json
from pathlib import Path

class BatchProcessor:
    """Process multiple datasets in batch."""
    
    def __init__(self, base_config: GeneratorConfig):
        self.base_config = base_config
        self.results = []
    
    def process_dataset(self, name: str, csv_path: str, output_suffix: str = None):
        """Process a single dataset."""
        print(f"Processing dataset: {name}")
        
        # Create dataset-specific configuration
        config = GeneratorConfig(
            templates_dir=self.base_config.templates_dir,
            csv_path=csv_path,
            output_dir=f"{self.base_config.output_dir}/{output_suffix or name}",
            image_dpi=self.base_config.image_dpi,
            enable_augmentations=self.base_config.enable_augmentations,
            max_workers=self.base_config.max_workers,
            batch_size=self.base_config.batch_size,
            strict_validation=self.base_config.strict_validation
        )
        
        try:
            generator = XFUNDGenerator(config)
            result = generator.generate_dataset()
            
            self.results.append({
                'name': name,
                'csv_path': csv_path,
                'output_dir': config.output_dir,
                'generated_entries': result.generated_entries,
                'failed_entries': result.failed_entries,
                'processing_time': result.processing_time,
                'status': 'success'
            })
            
            print(f"  ✓ Generated {result.generated_entries} entries")
            
        except Exception as e:
            self.results.append({
                'name': name,
                'csv_path': csv_path,
                'status': 'failed',
                'error': str(e)
            })
            print(f"  ✗ Failed: {e}")
    
    def process_all(self, datasets: dict):
        """Process all datasets."""
        print(f"Starting batch processing of {len(datasets)} datasets...")
        
        for name, csv_path in datasets.items():
            self.process_dataset(name, csv_path)
        
        self.save_batch_results()
        return self.results
    
    def save_batch_results(self):
        """Save batch processing results."""
        results_file = f"{self.base_config.output_dir}/batch_results.json"
        Path(self.base_config.output_dir).mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Batch results saved to: {results_file}")

# Usage example
base_config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="",  # Will be overridden
    output_dir="output/batch",
    image_dpi=600,
    enable_augmentations=True,
    max_workers=4,
    batch_size=10,
    strict_validation=True
)

datasets = {
    'medical_forms': 'data/csv/medical_data.csv',
    'insurance_claims': 'data/csv/insurance_data.csv',
    'tax_documents': 'data/csv/tax_data.csv',
    'legal_contracts': 'data/csv/legal_data.csv'
}

processor = BatchProcessor(base_config)
results = processor.process_all(datasets)

# Print summary
successful = [r for r in results if r['status'] == 'success']
failed = [r for r in results if r['status'] == 'failed']

print(f"\nBatch Processing Summary:")
print(f"  Successful: {len(successful)}")
print(f"  Failed: {len(failed)}")
print(f"  Total entries generated: {sum(r.get('generated_entries', 0) for r in successful)}")
```

### Parallel Processing

Process datasets in parallel for better performance:

```python
import concurrent.futures
from src.models import GeneratorConfig
from src.generate_dataset import XFUNDGenerator
import time

class ParallelProcessor:
    """Process datasets in parallel."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
    
    def process_single_dataset(self, dataset_info: dict) -> dict:
        """Process a single dataset (for parallel execution)."""
        name = dataset_info['name']
        config = GeneratorConfig(**dataset_info['config'])
        
        start_time = time.time()
        
        try:
            generator = XFUNDGenerator(config)
            result = generator.generate_dataset()
            
            return {
                'name': name,
                'status': 'success',
                'generated_entries': result.generated_entries,
                'failed_entries': result.failed_entries,
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'name': name,
                'status': 'failed',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def process_parallel(self, datasets: list) -> list:
        """Process multiple datasets in parallel."""
        print(f"Processing {len(datasets)} datasets in parallel (max_workers={self.max_workers})")
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_dataset = {
                executor.submit(self.process_single_dataset, dataset): dataset['name'] 
                for dataset in datasets
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_dataset):
                dataset_name = future_to_dataset[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'success':
                        print(f"✓ {dataset_name}: {result['generated_entries']} entries in {result['processing_time']:.2f}s")
                    else:
                        print(f"✗ {dataset_name}: {result['error']}")
                        
                except Exception as e:
                    print(f"✗ {dataset_name}: Unexpected error - {e}")
                    results.append({
                        'name': dataset_name,
                        'status': 'failed',
                        'error': f"Unexpected error: {e}"
                    })
        
        return results

# Usage example
datasets_parallel = [
    {
        'name': 'medical_forms_small',
        'config': {
            'templates_dir': 'data/templates_docx',
            'csv_path': 'data/csv/medical_small.csv',
            'output_dir': 'output/parallel/medical_small',
            'max_workers': 1,  # Single worker per dataset
            'batch_size': 5
        }
    },
    {
        'name': 'insurance_forms_small', 
        'config': {
            'templates_dir': 'data/templates_docx',
            'csv_path': 'data/csv/insurance_small.csv',
            'output_dir': 'output/parallel/insurance_small',
            'max_workers': 1,
            'batch_size': 5
        }
    }
]

processor = ParallelProcessor(max_workers=2)
results = processor.process_parallel(datasets_parallel)

print(f"\nParallel processing complete. Total time saved: ~{sum(r.get('processing_time', 0) for r in results)/2:.2f}s")
```

## Custom Data Sources

### Database Integration

Load data directly from databases:

```python
import sqlite3
import pandas as pd
from src.models import DataRecord
from typing import List

class DatabaseDataLoader:
    """Load XFUND data from database sources."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def load_from_sqlite(self, query: str) -> List[DataRecord]:
        """Load data from SQLite database."""
        conn = sqlite3.connect(self.db_path)
        
        try:
            df = pd.read_sql_query(query, conn)
            records = []
            
            for _, row in df.iterrows():
                record = DataRecord(
                    template_name=row['template_name'],
                    field_name=row['field_name'],
                    field_value=str(row['field_value']),
                    bbox=row['bbox']
                )
                records.append(record)
            
            return records
            
        finally:
            conn.close()
    
    def load_medical_forms(self, patient_id: str = None) -> List[DataRecord]:
        """Load medical form data with optional patient filtering."""
        query = """
        SELECT 
            'medical_form' as template_name,
            field_name,
            field_value,
            bbox_coordinates as bbox
        FROM medical_data
        """
        
        if patient_id:
            query += f" WHERE patient_id = '{patient_id}'"
        
        return self.load_from_sqlite(query)
    
    def load_by_date_range(self, start_date: str, end_date: str) -> List[DataRecord]:
        """Load data within date range."""
        query = f"""
        SELECT 
            template_name,
            field_name, 
            field_value,
            bbox
        FROM form_data 
        WHERE created_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY created_date
        """
        
        return self.load_from_sqlite(query)

# Usage example
db_loader = DatabaseDataLoader("data/forms.db")

# Load specific patient data
patient_data = db_loader.load_medical_forms(patient_id="P123456")

# Load data for date range
date_range_data = db_loader.load_by_date_range("2024-01-01", "2024-01-31")

print(f"Loaded {len(patient_data)} patient records")
print(f"Loaded {len(date_range_data)} date range records")
```

### API Integration

Load data from REST APIs:

```python
import requests
from src.models import DataRecord
from typing import List, Dict
import json

class APIDataLoader:
    """Load XFUND data from REST APIs."""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.headers = {}
        
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def load_from_api(self, endpoint: str, params: Dict = None) -> List[DataRecord]:
        """Load data from API endpoint."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        records = []
        
        for item in data.get('records', []):
            record = DataRecord(
                template_name=item['template_name'],
                field_name=item['field_name'],
                field_value=item['field_value'],
                bbox=item['bbox']
            )
            records.append(record)
        
        return records
    
    def load_paginated_data(self, endpoint: str, page_size: int = 100) -> List[DataRecord]:
        """Load all data with pagination."""
        all_records = []
        page = 1
        
        while True:
            params = {'page': page, 'page_size': page_size}
            
            try:
                page_records = self.load_from_api(endpoint, params)
                
                if not page_records:
                    break
                
                all_records.extend(page_records)
                page += 1
                
                print(f"Loaded page {page-1}: {len(page_records)} records")
                
            except requests.exceptions.RequestException as e:
                print(f"Error loading page {page}: {e}")
                break
        
        return all_records

# Usage example
api_loader = APIDataLoader("https://api.forms.example.com", api_key="your-api-key")

# Load specific dataset
form_data = api_loader.load_from_api("/forms/medical", params={'type': 'patient_intake'})

# Load all paginated data
all_data = api_loader.load_paginated_data("/forms/all")

print(f"Loaded {len(form_data)} form records")
print(f"Loaded {len(all_data)} total records")
```

## Advanced Augmentations

### Custom Augmentation Pipeline

Implement custom augmentation strategies:

```python
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from typing import Tuple, Optional

class AdvancedAugmentations:
    """Advanced image augmentation for document images."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def apply_realistic_scanner_effects(self, image: Image.Image) -> Image.Image:
        """Apply realistic scanner effects."""
        # Convert to numpy for OpenCV operations
        img_array = np.array(image)
        
        # Add slight skew
        if random.random() < 0.3:
            img_array = self._add_skew(img_array)
        
        # Add scanner noise
        if random.random() < 0.5:
            img_array = self._add_scanner_noise(img_array)
        
        # Add slight blur (scanner focus issues)
        if random.random() < 0.2:
            img_array = self._add_motion_blur(img_array)
        
        return Image.fromarray(img_array)
    
    def _add_skew(self, image: np.ndarray, max_angle: float = 2.0) -> np.ndarray:
        """Add slight skew to simulate scanner misalignment."""
        h, w = image.shape[:2]
        angle = random.uniform(-max_angle, max_angle)
        
        # Rotation matrix
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))
        return rotated
    
    def _add_scanner_noise(self, image: np.ndarray, intensity: float = 0.02) -> np.ndarray:
        """Add scanner-like noise."""
        noise = np.random.normal(0, intensity * 255, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def _add_motion_blur(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Add motion blur effect."""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        return cv2.filter2D(image, -1, kernel)
    
    def apply_document_aging(self, image: Image.Image) -> Image.Image:
        """Apply aging effects to documents."""
        # Slight yellowing
        if random.random() < 0.4:
            image = self._add_yellowing(image)
        
        # Faint stains or marks
        if random.random() < 0.2:
            image = self._add_stains(image)
        
        # Slight fading
        if random.random() < 0.3:
            image = self._add_fading(image)
        
        return image
    
    def _add_yellowing(self, image: Image.Image, intensity: float = 0.1) -> Image.Image:
        """Add slight yellowing effect."""
        enhancer = ImageEnhance.Color(image)
        # Reduce saturation slightly and shift toward yellow
        desaturated = enhancer.enhance(0.95)
        
        # Apply slight yellow tint
        yellow_overlay = Image.new('RGB', image.size, (255, 255, 240))
        return Image.blend(desaturated, yellow_overlay, intensity)
    
    def _add_stains(self, image: Image.Image) -> Image.Image:
        """Add subtle stain effects."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Random small stains
        for _ in range(random.randint(1, 3)):
            # Random position and size
            x = random.randint(0, w-20)
            y = random.randint(0, h-20)
            size = random.randint(5, 15)
            
            # Create subtle circular stain
            cv2.circle(img_array, (x, y), size, (240, 240, 240), -1)
        
        return Image.fromarray(img_array)
    
    def _add_fading(self, image: Image.Image, intensity: float = 0.1) -> Image.Image:
        """Add slight fading effect."""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1 + intensity)
    
    def apply_printing_effects(self, image: Image.Image) -> Image.Image:
        """Apply printing-related effects."""
        # Slight dot gain (darker text)
        if random.random() < 0.3:
            image = self._simulate_dot_gain(image)
        
        # Ink bleeding
        if random.random() < 0.2:
            image = self._simulate_ink_bleeding(image)
        
        return image
    
    def _simulate_dot_gain(self, image: Image.Image, intensity: float = 0.05) -> Image.Image:
        """Simulate printer dot gain effect."""
        img_array = np.array(image)
        
        # Dilate dark areas slightly
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(255 - gray, kernel, iterations=1)
        
        # Blend back
        result = 255 - dilated
        result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(result_rgb)
    
    def _simulate_ink_bleeding(self, image: Image.Image) -> Image.Image:
        """Simulate slight ink bleeding."""
        # Apply very subtle Gaussian blur
        return image.filter(ImageFilter.GaussianBlur(radius=0.5))

# Integration with XFUND Generator
class AugmentedXFUNDGenerator:
    """Extended generator with advanced augmentations."""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.augmenter = AdvancedAugmentations(seed=42)
    
    def generate_with_advanced_augmentations(self, image_path: str, output_path: str):
        """Generate image with advanced augmentations."""
        # Load original image
        image = Image.open(image_path)
        
        # Apply augmentations based on configuration
        if self.config.enable_augmentations:
            # Apply scanner effects
            image = self.augmenter.apply_realistic_scanner_effects(image)
            
            # Apply aging effects
            image = self.augmenter.apply_document_aging(image)
            
            # Apply printing effects  
            image = self.augmenter.apply_printing_effects(image)
        
        # Save augmented image
        image.save(output_path, dpi=(self.config.image_dpi, self.config.image_dpi))
        
        return output_path

# Usage example
config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="data/csv/data.csv",
    output_dir="output/advanced_augmented",
    enable_augmentations=True,
    image_dpi=600
)

generator = AugmentedXFUNDGenerator(config)

# Generate with advanced augmentations
augmented_path = generator.generate_with_advanced_augmentations(
    "input/document.png", 
    "output/advanced_augmented/document_aug.png"
)

print(f"Generated augmented image: {augmented_path}")
```

## Performance Optimization

### Memory-Efficient Processing

Handle large datasets with limited memory:

```python
import gc
import psutil
from src.models import GeneratorConfig
from src.generate_dataset import XFUNDGenerator
import time

class MemoryEfficientGenerator:
    """Memory-efficient processing for large datasets."""
    
    def __init__(self, config: GeneratorConfig, memory_limit_mb: int = 2000):
        self.config = config
        self.memory_limit_mb = memory_limit_mb
        self.chunk_size = self._calculate_optimal_chunk_size()
    
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available memory."""
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        
        # Conservative estimate: 100MB per image at 600 DPI
        memory_per_image = (self.config.image_dpi / 300) ** 2 * 25  # MB
        
        max_chunk = min(
            int(self.memory_limit_mb / memory_per_image),
            self.config.batch_size
        )
        
        return max(1, max_chunk)
    
    def process_in_chunks(self, data_records: list) -> dict:
        """Process data in memory-efficient chunks."""
        total_records = len(data_records)
        processed = 0
        results = {'generated': 0, 'failed': 0, 'errors': []}
        
        print(f"Processing {total_records} records in chunks of {self.chunk_size}")
        
        for i in range(0, total_records, self.chunk_size):
            chunk = data_records[i:i + self.chunk_size]
            chunk_size = len(chunk)
            
            print(f"Processing chunk {i//self.chunk_size + 1}: records {i+1}-{i+chunk_size}")
            
            # Monitor memory before processing
            memory_before = psutil.virtual_memory().percent
            
            try:
                # Create temporary configuration for chunk
                chunk_config = GeneratorConfig(
                    templates_dir=self.config.templates_dir,
                    csv_path=self._create_temp_csv(chunk),
                    output_dir=f"{self.config.output_dir}/chunk_{i//self.chunk_size}",
                    image_dpi=self.config.image_dpi,
                    enable_augmentations=self.config.enable_augmentations,
                    max_workers=min(self.config.max_workers, 2),  # Limit workers
                    batch_size=chunk_size,
                    strict_validation=self.config.strict_validation
                )
                
                # Process chunk
                generator = XFUNDGenerator(chunk_config)
                chunk_result = generator.generate_dataset()
                
                results['generated'] += chunk_result.generated_entries
                results['failed'] += chunk_result.failed_entries
                results['errors'].extend(chunk_result.errors)
                
                processed += chunk_size
                
                # Clean up
                del generator
                gc.collect()
                
                # Monitor memory after processing
                memory_after = psutil.virtual_memory().percent
                print(f"  Chunk complete: {chunk_result.generated_entries} generated, "
                      f"Memory: {memory_before:.1f}% -> {memory_after:.1f}%")
                
                # Check memory usage
                if memory_after > 85:
                    print("  High memory usage detected, forcing garbage collection")
                    gc.collect()
                    time.sleep(1)
                
            except Exception as e:
                print(f"  Chunk failed: {e}")
                results['errors'].append(f"Chunk {i//self.chunk_size}: {e}")
                results['failed'] += chunk_size
        
        print(f"\nMemory-efficient processing complete:")
        print(f"  Generated: {results['generated']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Errors: {len(results['errors'])}")
        
        return results
    
    def _create_temp_csv(self, records: list) -> str:
        """Create temporary CSV for chunk processing."""
        import tempfile
        import pandas as pd
        
        # Convert records to DataFrame
        data = []
        for record in records:
            data.append({
                'template_name': record.template_name,
                'field_name': record.field_name,
                'field_value': record.field_value,
                'bbox': record.bbox
            })
        
        df = pd.DataFrame(data)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        
        return temp_file.name

# Usage example
from src.utils import load_csv_data_as_models

# Load large dataset
large_dataset = load_csv_data_as_models("data/csv/large_dataset.csv")
print(f"Loaded {len(large_dataset)} records")

# Configure for memory efficiency
config = GeneratorConfig(
    templates_dir="data/templates_docx",
    csv_path="",  # Will be overridden
    output_dir="output/memory_efficient",
    image_dpi=300,  # Lower DPI for memory efficiency
    enable_augmentations=False,  # Disable to save memory
    max_workers=2,  # Limit workers
    batch_size=5,  # Small batches
    strict_validation=True
)

# Process with memory efficiency
efficient_generator = MemoryEfficientGenerator(config, memory_limit_mb=1500)
results = efficient_generator.process_in_chunks(large_dataset)
```

This advanced examples guide demonstrates sophisticated usage patterns for the XFUND Generator, including custom validation, batch processing, advanced augmentations, and memory-efficient processing for production environments.

---

*For more examples and use cases, see the other documentation files in the `docs/` directory.*