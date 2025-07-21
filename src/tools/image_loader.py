from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict
from PIL import Image
import logging
import base64
import io
from datetime import datetime
from PIL.ExifTags import TAGS

class ImageLoader:
    """Load and encode images from file system."""
    
    # Supported image formats
    SUPPORTED_FORMATS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff', '*.webp')
    
    def __init__(self, image_path: str) -> None:
        """
        Initialize ImageLoader with image path.
        
        Args:
            image_path: Path to single image file or directory containing images
        """
        self.image_path = Path(image_path)
        self._logger = logging.getLogger(__name__)

    def load_images_to_memory(self, single: bool = False) -> Union[Image.Image, List[Tuple[str, Image.Image]], None]:
        """
        Load images into memory.
        
        Args:
            single: If True, load single image. If False, load all images from directory.
            
        Returns:
            Single PIL Image if single=True, list of (name, image) tuples if single=False,
            None if no images found or error occurs.
            
        Raises:
            ValueError: If path doesn't exist or is invalid
            OSError: If image cannot be opened
        """
        if single:
            return self._load_single_image()
        else:
            return self._load_multiple_images()

    def _load_single_image(self) -> Optional[Image.Image]:
        """Load a single image file."""
        if not self.image_path.is_file():
            raise ValueError(f"Image file not found: {self.image_path}")
        
        try:
            # Open image first (required for EXIF extraction)
            image = Image.open(self.image_path)
            # Capture timestamp preferring EXIF metadata
            timestamp = self._extract_timestamp(image, self.image_path)
            image.info['timestamp'] = timestamp
            self._logger.info(f"Successfully loaded image: {self.image_path} (timestamp: {timestamp})")
            return image
        except Exception as e:
            self._logger.error(f"Failed to load image {self.image_path}: {e}")
            raise OSError(f"Cannot open image {self.image_path}: {e}")

    def _load_multiple_images(self) -> List[Tuple[str, Image.Image]]:
        """Load multiple images from directory."""
        if not self.image_path.is_dir():
            raise ValueError(f"Not a directory: {self.image_path}")
        
        images = []
        loaded_count = 0
        error_count = 0
        
        # Convert glob patterns to extensions for efficient lookup
        supported_extensions = {
            pattern.lower().replace('*', '') for pattern in self.SUPPORTED_FORMATS
        }
        
        # Walk directory tree only once and filter by extension
        for img_path in self.image_path.rglob('*'):
            if img_path.is_file() and img_path.suffix.lower() in supported_extensions:
                try:
                    image = Image.open(img_path)
                    # Store timestamp metadata on the image object (EXIF preferred)
                    timestamp = self._extract_timestamp(image, img_path)
                    image.info['timestamp'] = timestamp

                    images.append((img_path.name, image))
                    loaded_count += 1
                except Exception as e:
                    error_count += 1
                    self._logger.warning(f"Failed to load {img_path}: {e}")
                    continue
        
        self._logger.info(f"Loaded {loaded_count} images, {error_count} failed")
        return images

    def _extract_timestamp(self, image: Image.Image, path: Path) -> str:
        """Attempt to get creation timestamp from EXIF; fall back to filesystem timestamp."""
        # Try EXIF first
        try:
            exif_data = image.getexif()
            if exif_data:
                # Look for DateTimeOriginal (tag 36867) or DateTime (306)
                for tag_id in (36867, 36868, 306):
                    date_str = exif_data.get(tag_id)
                    if date_str:
                        # EXIF format: 'YYYY:MM:DD HH:MM:SS'
                        try:
                            dt = datetime.strptime(str(date_str), "%Y:%m:%d %H:%M:%S")
                            return dt.isoformat()
                        except ValueError:
                            # If parsing fails, continue to next tag/fallback
                            pass
        except Exception:
            pass

        # Fallback to filesystem timestamp (creation if available, else modification)
        stat_info = path.stat()
        return datetime.fromtimestamp(getattr(stat_info, 'st_ctime', stat_info.st_mtime)).isoformat()

    def resize_images(self, images: List[Tuple[str, Image.Image]], imwidth: int = 900, imheight: int = 900) -> List[Tuple[str, Image.Image]]:
        """Resize images to fit within the specified dimensions while maintaining aspect ratio."""
        resized_images: List[Tuple[str, Image.Image]] = []

        for name, image in images:
            w, h = image.size
            
            # Skip resizing if image is already smaller than target dimensions
            if w <= imwidth and h <= imheight:
                resized_images.append((name, image))
                continue

            # Calculate scaling ratio to fit within target dimensions
            scale_ratio = min(imwidth / w, imheight / h)
            
            # Only downscale, never upscale
            if scale_ratio >= 1.0:
                resized_images.append((name, image))
                continue
                
            # Calculate new dimensions maintaining aspect ratio
            new_width = int(w * scale_ratio)
            new_height = int(h * scale_ratio)
            
            # Resize with high-quality resampling
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_images.append((name, resized))
            self._logger.info(f"Resized {name} from {w}x{h} to {new_width}x{new_height}")

        return resized_images

    def encode_images(self, images: List[Tuple[str, Image.Image]], format: str = "PNG") -> List[Dict[str, str]]:
        """
        Encode images to base64 strings.
        
        Args:
            images: List of (name, PIL Image) tuples
            format: Image format for encoding (default: PNG)
            
        Returns:
            List of dictionaries with 'name' and 'data' keys containing base64 encoded images
        """
        encoded_images = []
        
        for name, image in images:
            try:
                # Convert to RGB if necessary (for JPEG compatibility)
                if format.upper() in ('JPEG', 'JPG') and image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                
                # Encode image
                buffered = io.BytesIO()
                image.save(buffered, format=format)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Retrieve previously stored timestamp (if any)
                timestamp = image.info.get('timestamp')
                encoded_images.append({
                    "name": name,
                    "data": img_str,
                    "timestamp": timestamp
                })
                
            except Exception as e:
                self._logger.error(f"Failed to encode image {name}: {e}")
                continue
        
        self._logger.info(f"Successfully encoded {len(encoded_images)} images")
        return encoded_images

    def encode_single_image(self, image: Image.Image, format: str = "PNG") -> str:
        """
        Encode a single image to base64 string.
        
        Args:
            image: PIL Image object
            format: Image format for encoding (default: PNG)
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Convert to RGB if necessary
            if format.upper() in ('JPEG', 'JPG') and image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            buffered = io.BytesIO()
            image.save(buffered, format=format)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
            
        except Exception as e:
            self._logger.error(f"Failed to encode image: {e}")
            raise OSError(f"Cannot encode image: {e}")

    def get_supported_formats(self) -> Tuple[str, ...]:
        """Get list of supported image formats."""
        return self.SUPPORTED_FORMATS 