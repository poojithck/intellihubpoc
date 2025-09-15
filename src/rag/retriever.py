import logging
from typing import List, Dict, Any

from .models import ReferenceImage, ImageCategory
from .repository import ReferenceImageRepository


class ImageRetriever:
    def __init__(self, repository: ReferenceImageRepository, config: Dict[str, Any]):
        self.repository = repository
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def retrieve_for_sor_type(self, sor_type: str, max_examples: int = None) -> Dict[str, List[ReferenceImage]]:
        """Retrieve reference images for a specific SOR type."""
        
        if max_examples is None:
            max_examples = self.config.get('max_reference_images', 3)
        
        self.logger.info(f"Retrieving reference images for {sor_type}")
        
        # Define retrieval logic per SOR type
        if sor_type == "MeterConsolidationE4":
            return self._retrieve_meter_consolidation_examples(max_examples)
        elif sor_type == "FuseReplacement":
            return self._retrieve_fuse_examples(max_examples)
        elif sor_type == "PlugInMeterRemoval":
            return self._retrieve_plug_meter_examples(max_examples)
        # Add more SOR types as needed
        else:
            return self._retrieve_default_examples(max_examples)
    
    def _retrieve_meter_consolidation_examples(self, max_examples: int) -> Dict[str, List[ReferenceImage]]:
        """Retrieve examples for meter consolidation."""
        examples = {
            'valid_meters': [],
            'not_meters': []
        }
        
        # Get valid meter examples
        valid_meters = self.repository.query_images(category=ImageCategory.VALID_METER)
        examples['valid_meters'] = valid_meters[:max_examples]
        
        # Get NOT a meter examples
        not_meters = self.repository.query_images(category=ImageCategory.NOT_A_METER)
        examples['not_meters'] = not_meters[:max_examples]
        
        total = len(examples['valid_meters']) + len(examples['not_meters'])
        self.logger.info(f"Retrieved {total} reference images ({len(examples['valid_meters'])} valid, {len(examples['not_meters'])} invalid)")
        
        return examples
    
    def _retrieve_fuse_examples(self, max_examples: int) -> Dict[str, List[ReferenceImage]]:
        """Retrieve examples for fuse replacement."""
        examples = {
            'valid_fuses': [],
            'not_valid_fuses': []  # Changed from 'invalid_fuses' to match your naming
        }
        
        # Get valid fuse examples
        valid_fuses = self.repository.query_images(category=ImageCategory.VALID_METER)
        examples['valid_fuses'] = valid_fuses[:max_examples]
        
        # Get NOT valid fuse examples (if any exist)
        not_valid_fuses = self.repository.query_images(category=ImageCategory.NOT_A_METER)
        examples['not_valid_fuses'] = not_valid_fuses[:max_examples]
        
        total = len(examples['valid_fuses']) + len(examples['not_valid_fuses'])
        self.logger.info(f"Retrieved {total} reference images ({len(examples['valid_fuses'])} valid, {len(examples['not_valid_fuses'])} not valid)")
        
        return examples
    
    def _retrieve_plug_meter_examples(self, max_examples: int) -> Dict[str, List[ReferenceImage]]:
        """Retrieve examples for plug-in meter removal."""
        examples = {
            'with_device': [],
            'without_device': []
        }
        
        # Customize for plug meter specific categories
        return examples
    
    def _retrieve_default_examples(self, max_examples: int) -> Dict[str, List[ReferenceImage]]:
        """Default retrieval for unknown SOR types."""
        return {}
    
    # Keep the original method for backward compatibility
    def retrieve_for_meter_consolidation(self, max_examples: int = None) -> Dict[str, List[ReferenceImage]]:
        return self.retrieve_for_sor_type("MeterConsolidationE4", max_examples)