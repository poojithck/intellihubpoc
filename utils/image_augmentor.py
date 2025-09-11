"""
Improved Electrical Meter Data Augmentation Pipeline with Debugging
===================================================================
Enhanced version with better error handling and verbose output.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json
import traceback

# Try importing required libraries with helpful error messages
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    print("Continuing without progress bars...")
    TQDM_AVAILABLE = False
    
    # Create a dummy tqdm for compatibility
    def tqdm(iterable, desc=None):
        if desc:
            print(f"{desc}...")
        return iterable

try:
    import albumentations as A
    from albumentations import Compose
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    print("ERROR: albumentations not installed!")
    print("Please install with: pip install albumentations")
    print("Or install all requirements: pip install opencv-python albumentations numpy tqdm")
    ALBUMENTATIONS_AVAILABLE = False

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('augmentation.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


class SimpleAugmentations:
    """Fallback augmentations using only OpenCV (no albumentations required)."""
    
    @staticmethod
    def brightness_contrast(image, brightness=0, contrast=0):
        """Adjust brightness and contrast."""
        brightness = int((brightness - 0.5) * 255)
        contrast = contrast + 1
        
        result = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return result
    
    @staticmethod
    def add_noise(image, noise_level=10):
        """Add gaussian noise."""
        noise = np.random.randn(*image.shape) * noise_level
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def blur(image, kernel_size=5):
        """Apply gaussian blur."""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def flip_horizontal(image):
        """Flip image horizontally."""
        return cv2.flip(image, 1)
    
    @staticmethod
    def rotate(image, angle):
        """Rotate image by given angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), 
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
        return rotated
    
    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        """Adjust gamma (brightness in non-linear way)."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    @staticmethod
    def add_shadow(image):
        """Add random shadow effect."""
        h, w = image.shape[:2]
        
        # Create random polygon for shadow
        points = np.array([
            [np.random.randint(0, w), 0],
            [np.random.randint(0, w), h],
            [np.random.randint(0, w), h],
            [np.random.randint(0, w), 0]
        ])
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        mask = cv2.GaussianBlur(mask, (51, 51), 20)
        
        # Apply shadow
        shadow_image = image.copy()
        shadow_image = cv2.convertScaleAbs(shadow_image, alpha=0.5, beta=0)
        
        result = np.where(mask[..., None] > 0,
                         (image * (255 - mask[..., None]) + shadow_image * mask[..., None]) / 255,
                         image).astype(np.uint8)
        return result


class MeterImageAugmenter:
    """Main augmentation class with fallback support."""
    
    def __init__(self, use_simple_augmentations=False):
        """
        Initialize augmenter.
        
        Args:
            use_simple_augmentations: Force use of simple OpenCV augmentations
        """
        self.use_simple = use_simple_augmentations or not ALBUMENTATIONS_AVAILABLE
        
        if self.use_simple:
            logger.info("Using simple OpenCV augmentations")
        else:
            logger.info("Using advanced albumentations library")
            self._setup_albumentations()
    
    def _setup_albumentations(self):
        """Setup albumentations pipelines."""
        try:
            # Light augmentation
            self.light_augmentation = Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            ])
            
            # Medium augmentation
            self.medium_augmentation = Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.Blur(blur_limit=3, p=0.4),
            ])
            
            # Heavy augmentation
            self.heavy_augmentation = Compose([
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.9),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, 
                             num_shadows_upper=2, shadow_dimension=5, p=0.5),
                A.GaussNoise(var_limit=(50.0, 100.0), p=0.7),
            ])
            
            # Geometric augmentation
            self.geometric_augmentation = Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.4),
            ])
            
            logger.info("Albumentations pipelines created successfully")
            
        except Exception as e:
            logger.error(f"Error setting up albumentations: {e}")
            self.use_simple = True
    
    def augment_image_simple(self, image: np.ndarray, aug_type: str = 'all') -> List[np.ndarray]:
        """Apply simple augmentations using only OpenCV."""
        augmented = []
        
        try:
            # Light augmentations
            if aug_type in ['light', 'all']:
                # Brightness variation
                aug = SimpleAugmentations.brightness_contrast(image, 0.7, 0.1)
                augmented.append(aug)
                
                # Slight blur
                aug = SimpleAugmentations.blur(image, 3)
                augmented.append(aug)
            
            # Medium augmentations
            if aug_type in ['medium', 'all']:
                # Contrast adjustment
                aug = SimpleAugmentations.brightness_contrast(image, 0.5, 0.3)
                augmented.append(aug)
                
                # Add noise
                aug = SimpleAugmentations.add_noise(image, 15)
                augmented.append(aug)
                
                # Gamma adjustment
                aug = SimpleAugmentations.adjust_gamma(image, 0.7)
                augmented.append(aug)
            
            # Heavy augmentations
            if aug_type in ['heavy', 'all']:
                # Strong brightness/contrast
                aug = SimpleAugmentations.brightness_contrast(image, 0.3, 0.4)
                augmented.append(aug)
                
                # Heavy noise
                aug = SimpleAugmentations.add_noise(image, 25)
                augmented.append(aug)
                
                # Add shadow
                aug = SimpleAugmentations.add_shadow(image)
                augmented.append(aug)
            
            # Geometric augmentations
            if aug_type in ['geometric', 'all']:
                # Horizontal flip
                aug = SimpleAugmentations.flip_horizontal(image)
                augmented.append(aug)
                
                # Small rotation
                aug = SimpleAugmentations.rotate(image, np.random.uniform(-5, 5))
                augmented.append(aug)
            
        except Exception as e:
            logger.error(f"Error in simple augmentation: {e}")
        
        return augmented if augmented else [image]
    
    def augment_image_advanced(self, image: np.ndarray, aug_type: str = 'all') -> List[np.ndarray]:
        """Apply advanced augmentations using albumentations."""
        augmented = []
        
        pipelines = []
        if aug_type == 'all':
            pipelines = [
                ('light', self.light_augmentation),
                ('medium', self.medium_augmentation),
                ('heavy', self.heavy_augmentation),
                ('geometric', self.geometric_augmentation)
            ]
        else:
            pipeline_map = {
                'light': self.light_augmentation,
                'medium': self.medium_augmentation,
                'heavy': self.heavy_augmentation,
                'geometric': self.geometric_augmentation
            }
            if aug_type in pipeline_map:
                pipelines = [(aug_type, pipeline_map[aug_type])]
        
        for name, pipeline in pipelines:
            try:
                result = pipeline(image=image)
                augmented.append(result['image'])
            except Exception as e:
                logger.warning(f"Augmentation {name} failed: {e}")
        
        return augmented if augmented else [image]
    
    def augment_image(self, image: np.ndarray, aug_type: str = 'all') -> List[np.ndarray]:
        """Apply augmentations to image."""
        if self.use_simple:
            return self.augment_image_simple(image, aug_type)
        else:
            return self.augment_image_advanced(image, aug_type)
    
    def process_folder(self, input_folder: str, output_folder: str, 
                      aug_type: str = 'all') -> Dict[str, Any]:
        """Process all images in folder."""
        
        logger.info("="*60)
        logger.info("STARTING AUGMENTATION PROCESS")
        logger.info("="*60)
        
        # Validate input folder
        input_path = Path(input_folder)
        if not input_path.exists():
            logger.error(f"Input folder does not exist: {input_folder}")
            print(f"\nERROR: Input folder '{input_folder}' does not exist!")
            print("Please check the path and try again.")
            return {'error': 'Input folder not found'}
        
        if not input_path.is_dir():
            logger.error(f"Input path is not a directory: {input_folder}")
            print(f"\nERROR: '{input_folder}' is not a directory!")
            return {'error': 'Input path is not a directory'}
        
        # Create output folder
        output_path = Path(output_folder)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output folder ready: {output_path}")
        except Exception as e:
            logger.error(f"Could not create output folder: {e}")
            print(f"\nERROR: Could not create output folder: {e}")
            return {'error': 'Could not create output folder'}
        
        # Find image files
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        logger.info(f"Searching for images in: {input_path}")
        for ext in extensions:
            found = list(input_path.glob(f'*{ext}')) + list(input_path.glob(f'*{ext.upper()}'))
            if found:
                logger.info(f"Found {len(found)} {ext} files")
            image_files.extend(found)
        
        if not image_files:
            logger.warning("No image files found!")
            print(f"\nWARNING: No image files found in '{input_folder}'")
            print(f"Supported formats: {', '.join(extensions)}")
            
            # List files in directory for debugging
            all_files = list(input_path.iterdir())
            if all_files:
                print(f"\nFiles found in directory:")
                for f in all_files[:10]:  # Show first 10 files
                    print(f"  - {f.name}")
                if len(all_files) > 10:
                    print(f"  ... and {len(all_files) - 10} more files")
            else:
                print("Directory is empty!")
            
            return {'error': 'No image files found'}
        
        print(f"\nFound {len(image_files)} image(s) to process")
        logger.info(f"Total images found: {len(image_files)}")
        
        # Create subdirectories
        subdirs = ['original']
        if aug_type == 'all':
            subdirs.extend(['light', 'medium', 'heavy', 'geometric'])
        else:
            subdirs.append(aug_type)
        
        for subdir in subdirs:
            (output_path / subdir).mkdir(exist_ok=True)
        
        # Process statistics
        stats = {
            'total_images': len(image_files),
            'processed': 0,
            'failed': 0,
            'augmented_created': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Process each image
        print("\nProcessing images...")
        
        iterator = tqdm(image_files, desc="Processing") if TQDM_AVAILABLE else image_files
        
        for img_path in iterator:
            try:
                logger.debug(f"Processing: {img_path.name}")
                
                # Read image
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.warning(f"Could not read image: {img_path.name}")
                    stats['failed'] += 1
                    continue
                
                # Save original
                original_output = output_path / 'original' / img_path.name
                cv2.imwrite(str(original_output), image)
                logger.debug(f"Saved original: {original_output.name}")
                
                # Convert to RGB for processing
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply augmentations
                augmented_images = self.augment_image(image_rgb, aug_type)
                logger.debug(f"Created {len(augmented_images)} augmentations")
                
                # Save augmented images
                for idx, aug_img in enumerate(augmented_images):
                    # Determine output directory
                    if aug_type == 'all':
                        # Map index to augmentation type
                        aug_dirs = ['light', 'medium', 'heavy', 'geometric']
                        dir_idx = idx % len(aug_dirs)
                        output_dir = aug_dirs[dir_idx]
                    else:
                        output_dir = aug_type
                    
                    # Convert back to BGR
                    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    
                    # Generate output filename
                    base_name = img_path.stem
                    output_name = f"{base_name}_aug_{idx:03d}.jpg"
                    output_file = output_path / output_dir / output_name
                    
                    # Save image
                    cv2.imwrite(str(output_file), aug_img_bgr, 
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
                    stats['augmented_created'] += 1
                    logger.debug(f"Saved augmented: {output_file.name}")
                
                stats['processed'] += 1
                
                if not TQDM_AVAILABLE and stats['processed'] % 5 == 0:
                    print(f"  Processed {stats['processed']}/{len(image_files)} images...")
                
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")
                logger.error(traceback.format_exc())
                stats['failed'] += 1
        
        # Save metadata
        metadata_path = output_path / 'augmentation_metadata.json'
        try:
            with open(metadata_path, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Metadata saved: {metadata_path}")
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
        
        # Print summary
        print("\n" + "="*60)
        print("AUGMENTATION COMPLETE")
        print("="*60)
        print(f"✓ Processed: {stats['processed']} images")
        print(f"✓ Augmented images created: {stats['augmented_created']}")
        print(f"✗ Failed: {stats['failed']} images")
        print(f"📁 Output location: {output_path.absolute()}")
        print("="*60)
        
        logger.info("Process completed")
        logger.info(f"Stats: {stats}")
        
        return stats


def main():
    """Main execution with command line support."""
    import argparse
    
    print("\n" + "="*60)
    print("METER IMAGE AUGMENTATION TOOL")
    print("="*60)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Augment electrical meter images')
    parser.add_argument('input_folder', help='Input folder containing images')
    parser.add_argument('output_folder', help='Output folder for augmented images')
    parser.add_argument('--type', default='all', 
                       choices=['light', 'medium', 'heavy', 'geometric', 'all'],
                       help='Type of augmentation (default: all)')
    parser.add_argument('--simple', action='store_true',
                       help='Use simple augmentations (no albumentations required)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Input folder: {args.input_folder}")
    print(f"  Output folder: {args.output_folder}")
    print(f"  Augmentation type: {args.type}")
    print(f"  Mode: {'Simple (OpenCV only)' if args.simple else 'Advanced'}")
    
    # Check dependencies
    if not args.simple and not ALBUMENTATIONS_AVAILABLE:
        print("\n⚠ WARNING: Albumentations not installed, using simple mode")
        print("  For better augmentations, install with:")
        print("  pip install albumentations")
    
    # Create augmenter
    try:
        augmenter = MeterImageAugmenter(use_simple_augmentations=args.simple)
    except Exception as e:
        print(f"\nERROR: Could not initialize augmenter: {e}")
        logger.error(f"Initialization error: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    # Process images
    try:
        stats = augmenter.process_folder(
            args.input_folder,
            args.output_folder,
            args.type
        )
        
        if 'error' in stats:
            return 1
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Unexpected error during processing: {e}")
        logger.error(f"Processing error: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())