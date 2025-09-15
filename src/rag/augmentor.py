import logging
from typing import List, Dict, Any

from .models import ReferenceImage, ImageCategory


class PromptAugmentor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def augment_prompt(
        self,
        base_system_prompt: str,
        base_main_prompt: str,
        reference_images: Dict[str, List[ReferenceImage]]
    ) -> tuple[str, str]:
        
        self.logger.info("Augmenting system prompt with reference examples")
        
        # Build minimal reference section for system prompt
        reference_section = self._build_minimal_reference_section(reference_images)
        
        # Augment ONLY the system prompt with reference examples
        augmented_system_prompt = base_system_prompt + "\n\n" + reference_section
        
        # Keep main prompt unchanged
        augmented_main_prompt = base_main_prompt
        
        self.logger.info(f"Added {sum(len(imgs) for imgs in reference_images.values())} reference examples to system prompt")
        
        return augmented_system_prompt, augmented_main_prompt
    
    def _build_minimal_reference_section(self, reference_images: Dict[str, List[ReferenceImage]]) -> str:
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
        
        if is_fuse_context:
            total_refs = len(reference_images.get('valid_fuses', [])) + len(reference_images.get('not_valid_fuses', []))
        else:
            total_refs = len(reference_images.get('valid_meters', [])) + len(reference_images.get('not_meters', []))
        
        self.logger.info(f"Combined {total_refs} reference + {len(work_order_images)} work order images")
        
        return combined