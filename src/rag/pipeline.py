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
        
        # Store per-SOR repositories
        self.repositories: Dict[str, ReferenceImageRepository] = {}
        self.retrievers: Dict[str, ImageRetriever] = {}
        
        # Single augmentor for all SOR types
        self.augmentor = PromptAugmentor(self.rag_config)
        
        # Initialize repositories for configured SOR types
        self._initialize_sor_repositories()
    
    def _load_rag_config(self) -> Dict[str, Any]:
        try:
            return self.config_manager.get_config("rag_config", subdirectory="rag")
        except Exception:
            self.logger.info("Using default RAG configuration")
            return {
                'enabled': True,
                'repository_path': 'data/reference_images',
                'max_reference_images': 50,
                'reference_images': {
                    'max_width': 800,
                    'max_height': 800,
                    'quality': 85
                },
                'enabled_sor_types': ['MeterConsolidationE4', 'FuseReplacement']
            }
    
    def _initialize_sor_repositories(self):
        """Initialize repositories for each configured SOR type."""
        enabled_sors = self.rag_config.get('enabled_sor_types', ['MeterConsolidationE4'])
        base_path = self.rag_config.get('repository_path', 'data/reference_images')
        
        for sor_type in enabled_sors:
            repository = ReferenceImageRepository(
                base_path=base_path,
                sor_type=sor_type
            )
            self.repositories[sor_type] = repository
            self.retrievers[sor_type] = ImageRetriever(repository, self.rag_config)
            self.logger.info(f"Initialized RAG repository for {sor_type} with {len(repository.images)} images")
    
    def enhance_sor_analysis(
        self,
        sor_type: str,
        prompt_config: Dict[str, Any],
        work_order_images: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Enhance SOR analysis by adding reference images to the main prompt.
        
        IMPORTANT: This method now returns ONLY the enhanced prompt configuration.
        Work order images are NOT modified or returned - they are handled separately by the bedrock client.
        
        Args:
            sor_type: Type of SOR analysis
            prompt_config: Original prompt configuration
            work_order_images: Work order images (used only for context, not modified)
            
        Returns:
            Enhanced prompt configuration with reference images in main prompt
        """
        
        if not self.rag_config.get('enabled', True):
            self.logger.info("RAG pipeline disabled")
            return prompt_config
        
        if not work_order_images:
            self.logger.info(f"No work order images provided for {sor_type}, skipping RAG enhancement")
            return prompt_config
        
        # Check if this SOR type has RAG enabled
        if sor_type not in self.repositories:
            self.logger.info(f"RAG not configured for {sor_type}")
            return prompt_config
        
        self.logger.info(f"Enhancing {sor_type} analysis with RAG pipeline")
        
        # Get the specific retriever for this SOR type
        retriever = self.retrievers[sor_type]
        
        # Retrieve relevant reference images
        max_refs = self.rag_config.get('max_reference_images', 50)
        reference_images = retriever.retrieve_for_sor_type(
            sor_type=sor_type,
            max_examples=max_refs
        )
        
        # Log retrieval results
        total_refs = sum(len(imgs) for imgs in reference_images.values())
        self.logger.info(f"Retrieved {total_refs} reference images for {sor_type}")
        
        # Augment the prompts - reference images now go to MAIN prompt
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
        enhanced_prompt_config['reference_count'] = total_refs
        
        self.logger.info(f"Enhanced main prompt with {total_refs} reference images")
        
        # Return ONLY the enhanced prompt config - work order images remain untouched
        return enhanced_prompt_config
    
    def get_repository(self, sor_type: str) -> Optional[ReferenceImageRepository]:
        """Get repository for a specific SOR type."""
        return self.repositories.get(sor_type)
    
    def add_reference_image(
        self,
        image_path: str,
        category: str,
        sor_type: str,
        image_id: Optional[str] = None
    ) -> ReferenceImage:
        """Add a reference image to a specific SOR repository."""
        
        if sor_type not in self.repositories:
            raise ValueError(f"No repository initialized for SOR type: {sor_type}")
        
        repository = self.repositories[sor_type]
        
        # Create metadata
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
        
        return repository.add_image(
            image_path,
            image_metadata,
            image_id,
            resize_config
        )
    
    def get_statistics(self, sor_type: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a specific SOR type or all."""
        from .models import ImageCategory
        
        if sor_type:
            if sor_type not in self.repositories:
                return {'error': f'No repository for {sor_type}'}
            
            repository = self.repositories[sor_type]
            all_images = repository.get_all_images()
        else:
            all_images = []
            for repository in self.repositories.values():
                all_images.extend(repository.get_all_images())
        
        stats = {
            'total_images': len(all_images),
            'valid_meters': 0,
            'not_meters': 0,
            'valid_fuses': 0,
            'not_valid_fuses': 0
        }
        
        for img in all_images:
            if img.metadata.category == ImageCategory.VALID_METER:
                stats['valid_meters'] += 1
            elif img.metadata.category == ImageCategory.NOT_A_METER:
                stats['not_meters'] += 1
            elif img.metadata.category == ImageCategory.VALID_FUSE:
                stats['valid_fuses'] += 1
            elif img.metadata.category == ImageCategory.NOT_VALID_FUSE:
                stats['not_valid_fuses'] += 1
        
        if not sor_type:
            stats['repositories'] = list(self.repositories.keys())
        
        return stats