from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import math
import logging

from .image_loader import ImageLoader
from ..config import ConfigManager

class ImageGridder:
    """
    Arranges images from a directory into grid images for LLM analysis.
    Grids are configurable (size, images per grid, borders, labels, etc).
    """
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize ImageGridder with configuration from config_manager.
        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.grid_config = self._get_grid_config()

    def _get_grid_config(self) -> dict:
        """
        Retrieve grid configuration from app config.
        Returns:
            Dictionary with grid configuration parameters
        """
        app_config = self.config_manager.get_app_config()
        return app_config.get("image_gridder", {
            "images_per_grid": 4,
            "grid_width": 1800,
            "grid_height": 1800,
            "border_size": 20,
            "background_color": "#FFFFFF",
            "label_images": True
        })

    def create_grids(self, image_dir: str, output_dir: Optional[str] = None) -> List[Tuple[str, Image.Image]]:
        """
        Load images from a directory and arrange them into grid images.
        Args:
            image_dir: Directory containing images
            output_dir: Optional directory to save grid images as PNGs
        Returns:
            List of (grid_name, grid_image) tuples
        """
        loader = ImageLoader(image_dir)
        images = loader.load_images_to_memory(single=False)
        if not images:
            self.logger.error(f"No images found in directory: {image_dir}")
            return []

        images_per_grid = self.grid_config["images_per_grid"]
        grid_width = self.grid_config["grid_width"]
        grid_height = self.grid_config["grid_height"]
        border_size = self.grid_config["border_size"]
        background_color = self.grid_config["background_color"]
        label_images = self.grid_config.get("label_images", True)

        # Calculate grid shape (e.g., 2x2, 3x3)
        grid_side = math.ceil(math.sqrt(images_per_grid))
        font = None
        label_height = 0
        if label_images:
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except Exception:
                font = ImageFont.load_default()
            # Estimate label height using a sample string
            label_height = font.getbbox("Sample")[3] - font.getbbox("Sample")[1] + 6  # add padding
        cell_width = (grid_width - (grid_side + 1) * border_size) // grid_side
        cell_height = (grid_height - (grid_side + 1) * border_size) // grid_side
        image_height = cell_height - label_height if label_images else cell_height

        grids = []
        total_images = len(images)
        num_grids = math.ceil(total_images / images_per_grid)

        for grid_idx in range(num_grids):
            grid_img = Image.new("RGB", (grid_width, grid_height), background_color)
            draw = ImageDraw.Draw(grid_img)

            for i in range(images_per_grid):
                img_idx = grid_idx * images_per_grid + i
                if img_idx >= total_images:
                    break
                name, img = images[img_idx]
                # Resize image to fit cell
                img = img.copy()
                img.thumbnail((cell_width, image_height), Image.Resampling.LANCZOS)
                # Calculate position
                row = i // grid_side
                col = i % grid_side
                x = border_size + col * (cell_width + border_size)
                y = border_size + row * (cell_height + border_size)
                # Paste image
                grid_img.paste(img, (x, y))
                # Draw label
                if label_images:
                    label_y = y + img.height + 2
                    label_x = x
                    # Draw a white rectangle behind the text for readability
                    text_bbox = font.getbbox(name)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    draw.rectangle([label_x, label_y, label_x + text_w + 4, label_y + text_h + 4], fill=background_color)
                    draw.text((label_x + 2, label_y + 2), name, fill="black", font=font)
            grid_name = f"grid_{grid_idx+1:02d}.png"
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                save_path = Path(output_dir) / grid_name
                grid_img.save(save_path)
                self.logger.info(f"Saved grid image to {save_path}")
            grids.append((grid_name, grid_img))
        self.logger.info(f"Created {len(grids)} grid images from {total_images} input images.")
        return grids

    def encode_grids(self, grids: List[Tuple[str, Image.Image]], format: str = "PNG") -> List[dict]:
        """
        Encode grid images to base64 for LLM input.
        Args:
            grids: List of (name, PIL.Image) tuples
            format: Image format for encoding (default: PNG)
        Returns:
            List of dicts with 'name', 'data', and 'timestamp' (None)
        """
        from .image_loader import ImageLoader
        encoded_grids = []
        for grid_name, grid_img in grids:
            encoded_data = ImageLoader.encode_single_image(grid_img, format=format)
            encoded_grids.append({
                "name": grid_name,
                "data": encoded_data,
                "timestamp": None
            })
        self.logger.info(f"Encoded {len(encoded_grids)} grid images for LLM input")
        return encoded_grids 