from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

from .image_loader import ImageLoader
from ..clients.bedrock_client import Bedrock_Client
from ..config import ConfigManager

class ImageAnalyzer:
    """Generic image analyzer using Bedrock AI with configurable prompts and parsing."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the image analyzer with configuration."""
        self.config_manager = config_manager
        self.bedrock_client = Bedrock_Client.from_config(config_manager)
        self.fallback_parser: Optional[Callable[[str], Dict[str, Any]]] = None
        self.logger = logging.getLogger(__name__)
    
    def set_fallback_parser(self, parser: Callable[[str], Dict[str, Any]]) -> None:
        """Set a custom fallback parser for non-JSON responses."""
        self.fallback_parser = parser
    
    def load_and_process_images(self, image_folder: str) -> List[Dict[str, str]]:
        """Load, resize, and encode images from the specified folder."""
        self.logger.info(f"Loading images from: {image_folder}")
        
        # Get image processing configuration
        img_config = self.config_manager.get_image_processing_config()
        
        # Initialize image loader
        loader = ImageLoader(image_folder)
        
        # Load images to memory
        images = loader.load_images_to_memory(single=False)
        if not images:
            self.logger.error("No images found in the specified folder")
            return []
        
        # Type check to ensure we have a list of tuples
        if not isinstance(images, list):
            self.logger.error("Expected list of images, got different type")
            return []
        
        self.logger.info(f"Loaded {len(images)} images")
        
        # Resize images using config values
        resize_config = img_config.get("default_resize", {"width": 900, "height": 900})
        resized_images = loader.resize_images(
            images, 
            imwidth=resize_config["width"], 
            imheight=resize_config["height"]
        )
        self.logger.info(f"Resized {len(resized_images)} images")
        
        # Encode images using config format
        format_config = img_config.get("default_format", "PNG")
        encoded_images = loader.encode_images(resized_images, format=format_config)
        self.logger.info(f"Encoded {len(encoded_images)} images")
        
        return encoded_images
    
    def _create_error_result(self, image_name: str, error_message: str) -> Dict[str, Any]:
        """Create a standardized error result."""
        return self.bedrock_client.create_analysis_result(
            image_name=image_name,
            parsed_response={"answer": "Error", "note": error_message},
            usage_info={},
            status="error"
        )
    
    async def analyze_image_async(self, image_data: Dict[str, str], prompt: str, 
                                model_params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single image asynchronously."""
        try:
            self.logger.info(f"Analyzing image: {image_data['name']}")
            
            # Send to Bedrock client using configured parameters
            response = self.bedrock_client.invoke_model(
                prompt=prompt,
                max_tokens=model_params.get("max_tokens", 200),
                temperature=model_params.get("temperature", 0.1),
                images=[image_data]
            )
            
            # Extract the result from the multimodal response
            if response.get("results") and len(response["results"]) > 0:
                result = response["results"][0]
                
                # Use the Bedrock client's JSON parsing method
                if "raw_text" in result:
                    parsed_response = self.bedrock_client.parse_json_response(
                        result["raw_text"], 
                        self.fallback_parser
                    )
                else:
                    parsed_response = result
                
                # Create standardized result using Bedrock client method
                return self.bedrock_client.create_analysis_result(
                    image_name=image_data["name"],
                    parsed_response=parsed_response,
                    usage_info=result,
                    status="success"
                )
                
            else:
                self.logger.error(f"No results returned for {image_data['name']}")
                return self._create_error_result(image_data["name"], "No response from model")
                
        except Exception as e:
            self.logger.error(f"Error analyzing {image_data['name']}: {e}")
            return self._create_error_result(image_data["name"], f"Analysis failed: {str(e)}")
    
    async def analyze_images_batch(self, encoded_images: List[Dict[str, str]], 
                                 prompt: str, model_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze multiple images concurrently."""
        self.logger.info(f"Starting batch analysis of {len(encoded_images)} images")
        
        # Create tasks for concurrent processing
        tasks = [
            self.analyze_image_async(image_data, prompt, model_params) 
            for image_data in encoded_images
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Task failed with exception: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def display_results(self, results: List[Dict[str, Any]], analysis_type: str = "Analysis") -> None:
        """Display analysis results in a readable format."""
        print("\n" + "="*80)
        print(f"{analysis_type.upper()} RESULTS")
        print("="*80)
        
        total_cost = 0.0
        status_counts = {}
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Image: {result['image_name']}")
            print(f"   Status: {result.get('status', 'Unknown')}")
            
            # Display main fields (answer, note, etc.)
            for key, value in result.items():
                if key not in ['image_name', 'status', 'input_tokens', 'output_tokens', 
                             'input_cost', 'output_cost']:
                    print(f"   {key.title()}: {value}")
            
            print(f"   Tokens: {result.get('input_tokens', 0)} input, {result.get('output_tokens', 0)} output")
            print(f"   Cost: ${result.get('input_cost', 0) + result.get('output_cost', 0):.4f}")
            
            # Count statuses
            status = result.get('status', 'Unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            total_cost += result.get('input_cost', 0) + result.get('output_cost', 0)
        
        # Summary
        print("\n" + "-"*80)
        print("SUMMARY")
        print("-"*80)
        print(f"Total images analyzed: {len(results)}")
        for status, count in status_counts.items():
            print(f"{status.title()}: {count}")
        print(f"Total cost: ${total_cost:.4f}")
        print("="*80)


class ConfigurableAnalyzer(ImageAnalyzer):
    """Configurable analyzer that can be used with any prompt configuration."""
    
    def __init__(self, config_manager: ConfigManager, analysis_type: str):
        """
        Initialize analyzer with a specific analysis configuration.
        
        Args:
            config_manager: ConfigManager instance
            analysis_type: Name of the analysis type (matches config file name)
        """
        super().__init__(config_manager)
        self.analysis_type = analysis_type
        
        # Load analysis-specific configuration
        self.prompt_config = config_manager.get_prompt_config(analysis_type)
        self.model_params = config_manager.get_model_params(
            config_type=self.prompt_config.get("model_config", "analysis")
        )
    
    def create_fallback_parser(self, keywords_config: Dict[str, List[str]]) -> Callable[[str], Dict[str, str]]:
        """Create a fallback parser based on keyword configuration."""
        def parser(text: str) -> Dict[str, str]:
            text = text.strip().lower()
            
            positive_keywords = keywords_config.get("positive", ["yes"])
            negative_keywords = keywords_config.get("negative", ["no"])
            
            # Look for indicators
            if any(keyword in text for keyword in positive_keywords):
                answer = "Yes"
            elif any(keyword in text for keyword in negative_keywords):
                answer = "No"
            else:
                answer = "Unknown"
            
            # Extract a note (first sentence)
            sentences = text.split('.')
            note = sentences[0] if sentences else "No note provided"
            
            return {"answer": answer, "note": note}
        
        return parser
    
    def setup_fallback_parser(self) -> None:
        """Setup fallback parser from configuration."""
        if "fallback_keywords" in self.prompt_config:
            parser = self.create_fallback_parser(self.prompt_config["fallback_keywords"])
            self.set_fallback_parser(parser)
    
    async def analyze_images(self, image_folder: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Analyze images using the configured analysis type.
        
        Args:
            image_folder: Optional folder path. If None, uses configured default.
            
        Returns:
            List of analysis results
        """
        # Setup fallback parser
        self.setup_fallback_parser()
        
        # Use configured default if no folder specified
        if image_folder is None:
            app_config = self.config_manager.get_app_config()
            image_folder = app_config.get("paths", {}).get("default_image_folder", "artefacts")
        
        # At this point image_folder is guaranteed to be a string
        folder_path: str = image_folder or "artefacts"
        
        # Ensure we have a valid folder path
        if not Path(folder_path).exists():
            self.logger.error(f"Image folder not found: {folder_path}")
            return []
        
        # Load and process images
        encoded_images = self.load_and_process_images(folder_path)
        if not encoded_images:
            self.logger.error("No images to analyze")
            return []
        
        # Analyze images using configured prompt and parameters
        results = await self.analyze_images_batch(
            encoded_images, 
            self.prompt_config["main_prompt"], 
            self.model_params
        )
        
        return results


class FuseAnalyzer(ConfigurableAnalyzer):
    """Specialized analyzer for fuse cartridge analysis."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the fuse analyzer."""
        super().__init__(config_manager, "fuse_analysis")
    
    async def analyze_fuse_images(self, image_folder: Optional[str] = None) -> List[Dict[str, Any]]:
        """Analyze fuse images from the specified folder."""
        return await self.analyze_images(image_folder) 