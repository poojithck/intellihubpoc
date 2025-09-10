import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import base64
from PIL import Image
import io

from .models import ReferenceImage, ImageMetadata, ImageCategory


class ReferenceImageRepository:
    def __init__(self, base_path: str = "data/reference_images", index_file: str = "index.json"):
        self.base_path = Path(base_path)
        self.index_file = self.base_path / index_file
        self.logger = logging.getLogger(__name__)
        self.images: Dict[str, ReferenceImage] = {}
        self._ensure_directories()
        self._load_index()
    
    def _ensure_directories(self):
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "meter_consolidation").mkdir(exist_ok=True)
    
    def _load_index(self):
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    for image_data in data.get('images', []):
                        ref_image = ReferenceImage.from_dict(image_data)
                        self.images[ref_image.id] = ref_image
                self.logger.info(f"Loaded {len(self.images)} reference images from index")
            except Exception as e:
                self.logger.error(f"Failed to load index: {e}")
                self.images = {}
        else:
            self.logger.info("No existing index found, starting with empty repository")
    
    def save_index(self):
        index_data = {
            'version': '1.0',
            'images': [img.to_dict() for img in self.images.values()]
        }
        with open(self.index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
        self.logger.info(f"Saved {len(self.images)} reference images to index")
    
    def add_image(self, image_path: str, metadata: ImageMetadata, 
                  image_id: Optional[str] = None, 
                  resize_config: Optional[Dict[str, Any]] = None) -> ReferenceImage:
        
        if image_id is None:
            image_id = f"ref_{len(self.images):04d}"
        
        full_path = Path(image_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load and encode image with resizing
        base64_data = self._encode_image(full_path, resize_config)
        
        ref_image = ReferenceImage(
            id=image_id,
            file_path=full_path,
            metadata=metadata,
            base64_data=base64_data
        )
        
        self.images[image_id] = ref_image
        self.save_index()
        self.logger.info(f"Added reference image: {image_id}")
        return ref_image
    
    def _encode_image(self, image_path: Path, 
                     resize_config: Optional[Dict[str, Any]] = None) -> str:
        
        # Default resize configuration
        if resize_config is None:
            resize_config = {
                'max_width': 800,
                'max_height': 800,
                'quality': 85
            }
        
        with Image.open(image_path) as img:
            # Apply resizing based on configuration
            max_width = resize_config.get('max_width', 800)
            max_height = resize_config.get('max_height', 800)
            
            if img.width > max_width or img.height > max_height:
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                self.logger.info(f"Resized image from {image_path.name} to {img.width}x{img.height}")
            
            buffer = io.BytesIO()
            img_format = 'JPEG' if image_path.suffix.lower() in ['.jpg', '.jpeg'] else 'PNG'
            quality = resize_config.get('quality', 85)
            
            img.save(buffer, format=img_format, optimize=True, quality=quality)
            buffer.seek(0)
            
            return base64.b64encode(buffer.read()).decode('utf-8')
    
    def get_image(self, image_id: str) -> Optional[ReferenceImage]:
        return self.images.get(image_id)
    
    def get_all_images(self) -> List[ReferenceImage]:
        return list(self.images.values())
    
    def query_images(
        self,
        category: Optional[ImageCategory] = None,
        shows_consolidation: Optional[bool] = None,
        min_confidence: float = 0.0
    ) -> List[ReferenceImage]:
        
        results = []
        for image in self.images.values():
            # Filter by category
            if category and image.metadata.category != category:
                continue
            
            # Filter by consolidation
            if shows_consolidation is not None and image.metadata.shows_consolidation != shows_consolidation:
                continue
            
            # Filter by confidence
            if image.metadata.confidence_score < min_confidence:
                continue
            
            results.append(image)
        
        return results
    
    def remove_image(self, image_id: str) -> bool:
        if image_id in self.images:
            del self.images[image_id]
            self.save_index()
            self.logger.info(f"Removed reference image: {image_id}")
            return True
        return False