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
    
    def augment_prompt(
        self,
        base_system_prompt: str,
        base_main_prompt: str,
        reference_images: Dict[str, List[ReferenceImage]]
    ) -> tuple[str, str]:
        """
        Augment prompts with reference examples.
        Now adds reference images to the MAIN prompt instead of system prompt.
        
        Args:
            base_system_prompt: Original system prompt
            base_main_prompt: Original main prompt
            reference_images: Dictionary of reference images by category
            
        Returns:
            Tuple of (system_prompt, augmented_main_prompt)
        """
        self.logger.info("Augmenting main prompt with reference examples")
        
        # Count total reference images
        total_refs = sum(len(imgs) for imgs in reference_images.values())
        
        # Build reference section with base64 encoded images
        reference_section = self._build_reference_section_with_images(reference_images)
        
        # Structure the augmented main prompt properly:
        # Original main prompt + Reference section + Work order analysis header
        augmented_main_prompt = (
            base_main_prompt + "\n\n" +
            reference_section + "\n\n" +
            "=" * 70 + "\n" +
            "WORK ORDER ANALYSIS" + "\n" +
            "=" * 70 + "\n" +
            "Now analyze the work order images below based on the reference examples provided above."
        )
        
        # Keep system prompt unchanged
        augmented_system_prompt = base_system_prompt
        
        self.logger.info(f"Added {total_refs} reference examples to main prompt")
        
        return augmented_system_prompt, augmented_main_prompt
    
    def _build_reference_section_with_images(self, reference_images: Dict[str, List[ReferenceImage]]) -> str:
        """Build reference section with base64 encoded images embedded in the prompt."""
        lines = []
        lines.append("=" * 70)
        lines.append("REFERENCE EXAMPLES FOR IDENTIFICATION")
        lines.append("=" * 70)
        lines.append("")
        lines.append("Study these reference examples carefully before analyzing the work order images.")
        lines.append("")
        
        # Check if we're dealing with fuses or meters
        is_fuse_context = 'valid_fuses' in reference_images or 'not_valid_fuses' in reference_images
        
        if is_fuse_context:
            # Handle fuse examples
            valid_fuses = reference_images.get('valid_fuses', [])
            if valid_fuses:
                lines.append("VALID FUSES - Examples to COUNT:")
                lines.append("These images show valid fuses that should be included in your count.")
                lines.append("")
                for i, img in enumerate(valid_fuses, 1):
                    if img.base64_data:
                        lines.append(f"Valid Fuse Example {i}:")
                        lines.append(f"<image>data:image/png;base64,{img.base64_data}</image>")
                        lines.append("")
            
            not_valid_fuses = reference_images.get('not_valid_fuses', [])
            if not_valid_fuses:
                lines.append("NOT VALID FUSES - Examples to EXCLUDE:")
                lines.append("These images show items that should NOT be counted as fuses.")
                lines.append("")
                for i, img in enumerate(not_valid_fuses, 1):
                    if img.base64_data:
                        lines.append(f"Not Valid Fuse Example {i}:")
                        lines.append(f"<image>data:image/png;base64,{img.base64_data}</image>")
                        lines.append("")
        else:
            # Handle meter examples
            valid_meters = reference_images.get('valid_meters', [])
            if valid_meters:
                lines.append("VALID METERS - Examples to COUNT:")
                lines.append("These images show valid meters that should be included in your count.")
                lines.append("")
                for i, img in enumerate(valid_meters, 1):
                    if img.base64_data:
                        lines.append(f"Valid Meter Example {i}:")
                        lines.append(f"<image>data:image/png;base64,{img.base64_data}</image>")
                        lines.append("")
            
            not_meters = reference_images.get('not_meters', [])
            if not_meters:
                lines.append("NON-METERS - Examples to EXCLUDE:")
                lines.append("These images show items that should NOT be counted as meters.")
                lines.append("")
                for i, img in enumerate(not_meters, 1):
                    if img.base64_data:
                        lines.append(f"Not A Meter Example {i}:")
                        lines.append(f"<image>data:image/png;base64,{img.base64_data}</image>")
                        lines.append("")
        
        lines.append("=" * 70)
        lines.append("Use the above reference examples to identify similar items in the work order images below.")
        lines.append("Apply the same identification criteria consistently across all work order images.")
        lines.append("=" * 70)
        
        return "\n".join(lines)