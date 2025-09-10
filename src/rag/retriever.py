import logging
from typing import List, Dict, Any

from .models import ReferenceImage, ImageCategory
from .repository import ReferenceImageRepository


class ImageRetriever:
    def __init__(self, repository: ReferenceImageRepository, config: Dict[str, Any]):
        self.repository = repository
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def retrieve_for_meter_consolidation(self, max_examples: int = None) -> Dict[str, List[ReferenceImage]]:
        
        if max_examples is None:
            max_examples = self.config.get('max_reference_images', 3)
        
        self.logger.info(f"Retrieving reference images for meter consolidation")
        
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