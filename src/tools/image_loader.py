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
    
    logger = logging.getLogger(__name__)

    def __init__(self, image_path: str, max_size_mb: float = 5.0) -> None:
        """
        Initialize ImageLoader with image path.
        
        Args:
            image_path: Path to single image file or directory containing images
            max_size_mb: Maximum allowed image size in MB (default: 5.0 for AWS compatibility)
        """
        self.image_path = Path(image_path)
        self.max_size_mb = max_size_mb
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

    def _apply_exif_orientation(self, image: Image.Image) -> Image.Image:
        """Apply EXIF orientation to the image if present."""
        try:
            exif = image.getexif()
            orientation_tag = 274  # ExifTags.ORIENTATION
            if exif and orientation_tag in exif:
                orientation = exif[orientation_tag]
                if orientation == 2:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 4:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                elif orientation == 5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 7:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        except Exception:
            pass
        return image

    def _load_single_image(self) -> Optional[Image.Image]:
        """Load a single image file."""
        if not self.image_path.is_file():
            raise ValueError(f"Image file not found: {self.image_path}")
        
        try:
            # Open image first (required for EXIF extraction)
            image = Image.open(self.image_path)
            image = self._apply_exif_orientation(image)
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
                    image = self._apply_exif_orientation(image)
                    # Store timestamp metadata on the image object (EXIF preferred)
                    timestamp = self._extract_timestamp(image, img_path)
                    image.info['timestamp'] = timestamp

                    images.append((img_path.name, image))
                    loaded_count += 1
                except Exception as e:
                    error_count += 1
                    self._logger.warning(f"Failed to load {img_path}: {e}")
                    continue
        
        # Sort images by timestamp (creation order) before returning
        images.sort(key=lambda x: x[1].info.get('timestamp', ''))
        
        self._logger.info(f"Loaded {loaded_count} images, {error_count} failed")
        self._logger.info(f"Images sorted by timestamp: {[name for name, _ in images]}")
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
        Encode images to base64 strings with automatic size limiting.
        Now uses the bulletproof encode_single_image() method.
        
        Args:
            images: List of (name, PIL Image) tuples
            format: Image format for encoding (default: PNG)
            
        Returns:
            List of dictionaries with 'name' and 'data' keys containing base64 encoded images
        """
        encoded_images = []
        
        for name, image in images:
            try:
                # Use the instance encode_single_image method which handles size limiting
                img_str = self.encode_single_image(image, format=format)
                
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
        Encode a single image to base64 string with automatic size limiting.
        Guarantees output is within the configured size limit.
        
        Args:
            image: PIL Image object
            format: Image format for encoding (default: PNG)
        Returns:
            Base64 encoded image string (guaranteed within size limit)
        Raises:
            OSError: If encoding fails
            ValueError: If image cannot be compressed below the configured limit
        """
        # Use instance variable for size limit
        MAX_ENCODED_MB = self.max_size_mb
        TARGET_MAX_RAW_MB = MAX_ENCODED_MB * 0.74  # ~74% of encoded size for raw bytes
        
        def _encode_to_b64(img: Image.Image) -> str:
            """Helper to encode PIL image to base64."""
            try:
                # Convert to RGB if necessary
                if format.upper() in ('JPEG', 'JPG') and img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                buffered = io.BytesIO()
                img.save(buffered, format=format, optimize=True)
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
            except Exception as e:
                ImageLoader.logger.error(f"Failed to encode image: {e}")
                raise OSError(f"Cannot encode image: {e}")
        
        # First try encoding as-is
        data = _encode_to_b64(image)
        encoded_size_mb = len(data.encode('utf-8')) / (1024 * 1024)
        raw_size_mb = len(base64.b64decode(data)) / (1024 * 1024)
        
        # Debug logging for large images
        if encoded_size_mb > MAX_ENCODED_MB:
            import traceback
            self._logger.error(
                f"LARGE IMAGE DETECTED: raw={raw_size_mb:.2f}MB, encoded={encoded_size_mb:.2f}MB - Size: {image.size}, Mode: {image.mode}")
            self._logger.error(f"Call stack: {traceback.format_stack()[-3:-1]}")
        else:
            self._logger.info(
                f"Image encoded: raw={raw_size_mb:.2f}MB, encoded={encoded_size_mb:.2f}MB - Size: {image.size}, Mode: {image.mode}")
        
        if encoded_size_mb <= MAX_ENCODED_MB:
            return data  # Fast path - image is already within limits
        
        self._logger.warning(
            f"Image encoded size is {encoded_size_mb:.2f}MB, exceeds {MAX_ENCODED_MB}MB limit. Auto-compressing...")
        
        # If target is JPEG, try quality reduction first (much faster than resampling)
        if format.upper() == 'JPEG':
            for quality in [95, 90, 85, 80, 75, 70, 65, 60, 55]:
                try:
                    jpeg_buffer = io.BytesIO()
                    rgb_img = image.convert('RGB') if image.mode != 'RGB' else image
                    rgb_img.save(jpeg_buffer, format='JPEG', quality=quality, optimize=True)
                    jpeg_b64 = base64.b64encode(jpeg_buffer.getvalue()).decode('utf-8')
                    encoded_jpeg_mb = len(jpeg_b64.encode('utf-8')) / (1024 * 1024)
                    if encoded_jpeg_mb <= MAX_ENCODED_MB:
                        self._logger.info(f"JPEG quality reduction: quality={quality}, encoded={encoded_jpeg_mb:.2f}MB")
                        return jpeg_b64
                except Exception:
                    continue
        
        # Progressive minimal-loss resize: 95%, 90%, 85%, 80%, 75%, 70%, 65%, 60%, 55%, 50%
        resize_factors = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
        encoded_final_mb = encoded_size_mb  # Initialize for error message
        raw_final_mb = raw_size_mb
        for factor in resize_factors:
            new_width = int(image.width * factor)
            new_height = int(image.height * factor)
            resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            data = _encode_to_b64(resized_img)
            encoded_final_mb = len(data.encode('utf-8')) / (1024 * 1024)
            raw_final_mb = len(base64.b64decode(data)) / (1024 * 1024)
            
            if encoded_final_mb <= MAX_ENCODED_MB:
                self._logger.info(
                    f"Auto-compressed image from encoded={encoded_size_mb:.2f}MB to encoded={encoded_final_mb:.2f}MB via {factor*100:.0f}% resize (raw {raw_final_mb:.2f}MB)")
                return data
        
        # Fallback 2 – convert to JPEG and vary quality (only if not already JPEG)
        if format.upper() != 'JPEG':
            for quality in [95, 90, 85, 80, 75, 70, 60, 50]:
                jpeg_buffer = io.BytesIO()
                rgb_img = image.convert('RGB') if image.mode != 'RGB' else image
                rgb_img.save(jpeg_buffer, format='JPEG', quality=quality, optimize=True)
                jpeg_b64 = base64.b64encode(jpeg_buffer.getvalue()).decode('utf-8')
                encoded_jpeg_mb = len(jpeg_b64.encode('utf-8')) / (1024 * 1024)
                if encoded_jpeg_mb <= MAX_ENCODED_MB:
                    self._logger.info(f"JPEG fallback succeeded: quality={quality}, encoded={encoded_jpeg_mb:.2f}MB")
                    return jpeg_b64
        
        # Last resort – raise explicit error
        raise ValueError(
            f"Image still encoded size {encoded_final_mb:.2f}MB after all compression steps (exceeds {MAX_ENCODED_MB}MB configured limit)")

    def get_supported_formats(self) -> Tuple[str, ...]:
        """Get list of supported image formats."""
        return self.SUPPORTED_FORMATS 