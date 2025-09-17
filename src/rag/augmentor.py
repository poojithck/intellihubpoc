import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import base64
import io
from PIL import Image, ImageDraw, ImageFont

from .models import ReferenceImage, ImageCategory


class PromptAugmentor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Create directory for saving gridded reference images
        self.grid_cache_dir = Path("data/reference_grids")
        self.grid_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_reference_grid(self, images: List[ReferenceImage], category: str, 
                               images_per_grid: int = 2) -> List[Tuple[str, str]]:
        """
        Create gridded images from reference images.
        
        Args:
            images: List of reference images
            category: Category name (e.g., 'valid_meters', 'not_meters')
            images_per_grid: Number of images per grid (default: 2)
            
        Returns:
            List of tuples (grid_name, base64_encoded_grid)
        """
        grids = []
        
        # Grid configuration
        grid_width = 1200  # Total width of grid
        grid_height = 600   # Total height for 2x1 grid
        border_size = 20    # White space between images
        background_color = (255, 255, 255)  # White background
        
        # Calculate cell dimensions
        cells_per_row = images_per_grid
        cells_per_col = 1
        cell_width = (grid_width - (cells_per_row + 1) * border_size) // cells_per_row
        cell_height = (grid_height - (cells_per_col + 1) * border_size) // cells_per_col
        
        # Process images in batches
        for grid_idx in range(0, len(images), images_per_grid):
            batch = images[grid_idx:grid_idx + images_per_grid]
            
            # Create new grid image
            grid_img = Image.new('RGB', (grid_width, grid_height), background_color)
            draw = ImageDraw.Draw(grid_img)
            
            # Try to load a font for labels
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            # Place each image in the grid
            for i, ref_img in enumerate(batch):
                try:
                    # Decode base64 image
                    img_data = base64.b64decode(ref_img.base64_data)
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Calculate position (horizontal layout)
                    col = i % cells_per_row
                    row = i // cells_per_row
                    x = border_size + col * (cell_width + border_size)
                    y = border_size + row * (cell_height + border_size)
                    
                    # Resize image to fit cell
                    img.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
                    
                    # Center image in cell
                    x_offset = (cell_width - img.width) // 2
                    y_offset = (cell_height - img.height) // 2
                    
                    # Paste image
                    grid_img.paste(img, (x + x_offset, y + y_offset))
                    
                    # Add label below image
                    label = f"{category}_{i+1}"
                    label_y = y + y_offset + img.height + 5
                    text_bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    label_x = x + (cell_width - text_width) // 2
                    draw.text((label_x, label_y), label, fill='black', font=font)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to add image to grid: {e}")
                    continue
            
            # Save grid locally
            grid_name = f"ref_grid_{category}_{grid_idx//images_per_grid:03d}"
            grid_path = self.grid_cache_dir / f"{grid_name}.jpg"
            grid_img.save(grid_path, 'JPEG', quality=95)
            self.logger.info(f"Saved reference grid: {grid_path}")
            
            # Encode grid to base64
            buffer = io.BytesIO()
            grid_img.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)
            grid_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            grids.append((grid_name, grid_base64))
        
        return grids
    
    def augment_prompt(
        self,
        base_system_prompt: str,
        base_main_prompt: str,
        reference_images: Dict[str, List[ReferenceImage]]
    ) -> tuple[str, str]:
        
        self.logger.info("Augmenting system prompt with reference examples")
        
        # Count total reference images
        total_refs = sum(len(imgs) for imgs in reference_images.values())
        
        # Decide whether to use gridding based on count
        use_gridding = total_refs > 15
        
        if use_gridding:
            self.logger.info(f"Total reference images ({total_refs}) exceeds 15, using gridded approach")
            reference_section = self._build_gridded_reference_section(reference_images)
        else:
            self.logger.info(f"Using individual reference images ({total_refs} images)")
            reference_section = self._build_minimal_reference_section(reference_images)
        
        # Augment ONLY the system prompt with reference examples
        augmented_system_prompt = base_system_prompt + "\n\n" + reference_section
        
        # Keep main prompt unchanged
        augmented_main_prompt = base_main_prompt
        
        self.logger.info(f"Added reference examples to system prompt (gridded: {use_gridding})")
        
        return augmented_system_prompt, augmented_main_prompt
    
    def _build_gridded_reference_section(self, reference_images: Dict[str, List[ReferenceImage]]) -> str:
        """Build reference section for gridded images."""
        lines = []
        lines.append("=" * 70)
        lines.append("VISUAL REFERENCE EXAMPLES (GRIDDED FORMAT)")
        lines.append("=" * 70)
        lines.append("")
        lines.append("IMPORTANT: Reference examples are presented in grid format.")
        lines.append("Each grid image contains 2 reference examples side by side.")
        lines.append("White borders separate individual examples within each grid.")
        lines.append("")
        
        # Check if we're dealing with fuses or meters
        is_fuse_context = 'valid_fuses' in reference_images or 'not_valid_fuses' in reference_images
        
        if is_fuse_context:
            # Handle fuse examples
            valid_fuses = reference_images.get('valid_fuses', [])
            if valid_fuses:
                num_grids = (len(valid_fuses) + 1) // 2  # Round up
                lines.append(f"VALID FUSES - {len(valid_fuses)} examples in {num_grids} grid(s):")
                lines.append("Each grid shows 2 valid fuse examples.")
                lines.append("COUNT fuses that resemble these examples.")
                for i in range(num_grids):
                    lines.append(f"[Grid {i+1}: REFERENCE_GRID_valid_fuses_{i:03d}]")
                lines.append("")
            
            not_valid_fuses = reference_images.get('not_valid_fuses', [])
            if not_valid_fuses:
                num_grids = (len(not_valid_fuses) + 1) // 2
                lines.append(f"NOT VALID FUSES - {len(not_valid_fuses)} examples in {num_grids} grid(s):")
                lines.append("Each grid shows 2 examples of items that are NOT valid fuses.")
                lines.append("DO NOT count items that resemble these examples.")
                for i in range(num_grids):
                    lines.append(f"[Grid {i+1}: REFERENCE_GRID_not_valid_fuses_{i:03d}]")
                lines.append("")
        else:
            # Handle meter examples
            valid_meters = reference_images.get('valid_meters', [])
            if valid_meters:
                num_grids = (len(valid_meters) + 1) // 2
                lines.append(f"VALID METERS - {len(valid_meters)} examples in {num_grids} grid(s):")
                lines.append("Each grid shows 2 valid meter examples.")
                lines.append("COUNT meters that resemble these examples.")
                for i in range(num_grids):
                    lines.append(f"[Grid {i+1}: REFERENCE_GRID_valid_meters_{i:03d}]")
                lines.append("")
            
            not_meters = reference_images.get('not_meters', [])
            if not_meters:
                num_grids = (len(not_meters) + 1) // 2
                lines.append(f"NON-METERS - {len(not_meters)} examples in {num_grids} grid(s):")
                lines.append("Each grid shows 2 examples of items that are NOT meters.")
                lines.append("DO NOT count items that resemble these examples.")
                for i in range(num_grids):
                    lines.append(f"[Grid {i+1}: REFERENCE_GRID_not_meters_{i:03d}]")
                lines.append("")
        
        lines.append("=" * 70)
        lines.append("ANALYSIS INSTRUCTIONS:")
        lines.append("1. Study each grid carefully - they contain multiple reference examples")
        lines.append("2. Compare work order images below against ALL reference examples")
        lines.append("3. Apply the same identification criteria consistently")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _build_minimal_reference_section(self, reference_images: Dict[str, List[ReferenceImage]]) -> str:
        """Original method for individual reference images (when count <= 15)."""
        lines = []
        lines.append("=" * 70)
        lines.append("VISUAL REFERENCE EXAMPLES")
        lines.append("=" * 70)
        
        # Check if we're dealing with fuses or meters based on the keys
        is_fuse_context = 'valid_fuses' in reference_images or 'not_valid_fuses' in reference_images
        
        if is_fuse_context:
            # Handle fuse examples
            valid_fuses = reference_images.get('valid_fuses', [])
            if valid_fuses:
                lines.append("\nEXAMPLES OF VALID FUSES (COUNT FUSES RESEMBLING THESE):")
                for i, img in enumerate(valid_fuses, 1):
                    lines.append(f"VALID FUSE Example {i}")
                    lines.append(f"[Image: REFERENCE_{i:02d}_valid_fuse]")
                    lines.append("")
            
            # Non-valid fuse examples
            not_valid_fuses = reference_images.get('not_valid_fuses', [])
            if not_valid_fuses:
                lines.append("\nEXAMPLES OF NOT VALID FUSES (DO NOT COUNT THESE):")
                for i, img in enumerate(not_valid_fuses, 1):
                    lines.append(f"NOT A VALID FUSE Example {i}")
                    lines.append(f"[Image: REFERENCE_{i:02d}_not_valid_fuse]")
                    lines.append("")
        else:
            # Handle meter examples (existing code)
            valid_meters = reference_images.get('valid_meters', [])
            if valid_meters:
                lines.append("\nEXAMPLES OF VALID METERS (COUNT METERS RESEMBLING THESE):")
                for i, img in enumerate(valid_meters, 1):
                    lines.append(f"VALID METER Example {i}")
                    lines.append(f"[Image: REFERENCE_{i:02d}_valid_meter]")
                    lines.append("")
            
            # Non-meter examples
            not_meters = reference_images.get('not_meters', [])
            if not_meters:
                lines.append("\nEXAMPLES OF NON-METERS (DO NOT COUNT THESE):")
                for i, img in enumerate(not_meters, 1):
                    lines.append(f"NOT A METER Example {i}")
                    lines.append(f"[Image: REFERENCE_{i:02d}_not_a_meter]")
                    lines.append("")
        
        lines.append("=" * 70)
        if is_fuse_context:
            lines.append("Use these visual examples to identify fuses in the work order images below.")
        else:
            lines.append("Use these visual examples to identify meters in the work order images below.")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def combine_images(
        self,
        reference_images: Dict[str, List[ReferenceImage]],
        work_order_images: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        
        combined = []
        
        # Count total reference images
        total_refs = sum(len(imgs) for imgs in reference_images.values())
        use_gridding = total_refs > 15
        
        if use_gridding:
            self.logger.info(f"Using gridded approach for {total_refs} reference images")
            
            # Create grids for each category
            for category, images in reference_images.items():
                if images:
                    grids = self._create_reference_grid(images, category, images_per_grid=2)
                    for grid_name, grid_data in grids:
                        combined.append({
                            'name': f"REFERENCE_GRID_{grid_name}",
                            'data': grid_data,
                            'media_type': 'image/jpeg',
                            'timestamp': None
                        })
                    self.logger.info(f"Created {len(grids)} grids for {category}")
        else:
            # Use original individual image approach
            ref_counter = 1
            
            # Check if we're dealing with fuses or meters
            is_fuse_context = 'valid_fuses' in reference_images or 'not_valid_fuses' in reference_images
            
            if is_fuse_context:
                # Add valid fuse examples
                for img in reference_images.get('valid_fuses', []):
                    if img.base64_data:
                        combined.append({
                            'name': f"REFERENCE_{ref_counter:02d}_valid_fuse",
                            'data': img.base64_data,
                            'media_type': 'image/jpeg',
                            'timestamp': None
                        })
                        ref_counter += 1
                
                # Reset counter for not_valid_fuse examples
                ref_counter = 1
                
                # Add not valid fuse examples
                for img in reference_images.get('not_valid_fuses', []):
                    if img.base64_data:
                        combined.append({
                            'name': f"REFERENCE_{ref_counter:02d}_not_valid_fuse",
                            'data': img.base64_data,
                            'media_type': 'image/jpeg',
                            'timestamp': None
                        })
                        ref_counter += 1
            else:
                # Handle meter examples (existing code)
                for img in reference_images.get('valid_meters', []):
                    if img.base64_data:
                        combined.append({
                            'name': f"REFERENCE_{ref_counter:02d}_valid_meter",
                            'data': img.base64_data,
                            'media_type': 'image/jpeg',
                            'timestamp': None
                        })
                        ref_counter += 1
                
                ref_counter = 1
                
                for img in reference_images.get('not_meters', []):
                    if img.base64_data:
                        combined.append({
                            'name': f"REFERENCE_{ref_counter:02d}_not_a_meter",
                            'data': img.base64_data,
                            'media_type': 'image/jpeg',
                            'timestamp': None
                        })
                        ref_counter += 1
        
        # Add work order images
        for img in work_order_images:
            combined.append({
                'name': f"WORKORDER_{img['name']}",
                'data': img['data'],
                'media_type': img.get('media_type', 'image/jpeg'),
                'timestamp': img.get('timestamp')
            })
        
        if use_gridding:
            grid_count = len([c for c in combined if 'GRID' in c['name']])
            self.logger.info(f"Combined {grid_count} reference grids + {len(work_order_images)} work order images")
        else:
            if is_fuse_context:
                total_refs = len(reference_images.get('valid_fuses', [])) + len(reference_images.get('not_valid_fuses', []))
            else:
                total_refs = len(reference_images.get('valid_meters', [])) + len(reference_images.get('not_meters', []))
            
            self.logger.info(f"Combined {total_refs} reference + {len(work_order_images)} work order images")
        
        return combined