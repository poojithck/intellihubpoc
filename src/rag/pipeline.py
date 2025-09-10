import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .repository import ReferenceImageRepository
from .retriever import ImageRetriever
from .augmentor import PromptAugmentor
from .models import ReferenceImage, ImageMetadata, ImageCategory


class RAGPipeline:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Load RAG configuration
        self.rag_config = self._load_rag_config()
        
        # Initialize components
        self.repository = ReferenceImageRepository(
            self.rag_config.get('repository_path', 'data/reference_images')
        )
        self.retriever = ImageRetriever(self.repository, self.rag_config)
        self.augmentor = PromptAugmentor(self.rag_config)
    
    def _load_rag_config(self) -> Dict[str, Any]:
        try:
            return self.config_manager.get_config("rag_config", subdirectory="rag")
        except Exception:
            self.logger.info("Using default RAG configuration")
            return {
                'enabled': True,
                'repository_path': 'data/reference_images',
                'max_reference_images': 3,
                'reference_images': {
                    'max_width': 800,
                    'max_height': 800,
                    'quality': 85
                }
            }
    
    def enhance_sor_analysis(
        self,
        sor_type: str,
        prompt_config: Dict[str, Any],
        work_order_images: List[Dict[str, str]]
    ) -> tuple[Dict[str, Any], List[Dict[str, str]]]:
        
        if not self.rag_config.get('enabled', True):
            self.logger.info("RAG pipeline disabled")
            return prompt_config, work_order_images
        
        if sor_type != "MeterConsolidationE4":
            self.logger.info(f"RAG not configured for {sor_type}")
            return prompt_config, work_order_images
        
        self.logger.info(f"Enhancing {sor_type} analysis with RAG pipeline")
        
        # Retrieve relevant reference images
        reference_images = self.retriever.retrieve_for_meter_consolidation(
            max_examples=self.rag_config.get('max_reference_images', 3)
        )
        
        # Augment the prompts
        base_system_prompt = prompt_config.get('system_prompt', '')
        base_main_prompt = prompt_config.get('main_prompt', '')
        
        augmented_system_prompt, augmented_main_prompt = self.augmentor.augment_prompt(
            base_system_prompt,
            base_main_prompt,
            reference_images
        )
        
        # Update prompt configuration
        enhanced_prompt_config = prompt_config.copy()
        enhanced_prompt_config['system_prompt'] = augmented_system_prompt
        enhanced_prompt_config['main_prompt'] = augmented_main_prompt
        enhanced_prompt_config['rag_enhanced'] = True
        
        # Combine images
        combined_images = self.augmentor.combine_images(reference_images, work_order_images)
        
        return enhanced_prompt_config, combined_images
    
    def add_reference_image(
        self,
        image_path: str,
        category: str,
        image_id: Optional[str] = None
    ) -> ReferenceImage:
        
        # Create simple metadata with just category
        image_metadata = ImageMetadata(
            category=ImageCategory(category),
            confidence_score=1.0
        )
        
        # Get resize configuration
        resize_config = self.rag_config.get('reference_images', {
            'max_width': 800,
            'max_height': 800,
            'quality': 85
        })
        
        return self.repository.add_image(
            image_path,
            image_metadata,
            image_id,
            resize_config
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        all_images = self.repository.get_all_images()
        
        stats = {
            'total_images': len(all_images),
            'valid_meters': 0,
            'not_meters': 0
        }
        
        for img in all_images:
            if img.metadata.category == ImageCategory.VALID_METER:
                stats['valid_meters'] += 1
            elif img.metadata.category == ImageCategory.NOT_A_METER:
                stats['not_meters'] += 1
        
        return stats