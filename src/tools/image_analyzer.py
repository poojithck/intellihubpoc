from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

from .image_loader import ImageLoader
from ..clients.bedrock_client import BedrockClient
from ..config import ConfigManager

class ImageAnalyzer:
    """Configurable image analyzer using Bedrock AI with configuration-based prompts and parsing."""
    
    def __init__(self, config_manager: ConfigManager, analysis_type: str, max_concurrent_requests: int = 10):
        """
        Initialize the image analyzer with configuration.
        Args:
            config_manager: ConfigManager instance
            analysis_type: Name of the analysis type (matches config file name)
            max_concurrent_requests: Maximum number of concurrent API requests (default: 10)
        """
        self.config_manager = config_manager
        self.analysis_type = analysis_type
        self.bedrock_client = BedrockClient.from_config(config_manager)
        self.fallback_parser: Optional[Callable[[str], Dict[str, Any]]] = None
        self.logger = logging.getLogger(__name__)
        
        # Create semaphore for limiting concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Load analysis-specific configuration
        self.prompt_config = config_manager.get_prompt_config(analysis_type)
        self.model_params = config_manager.get_model_params(
            config_type=self.prompt_config.get("model_config", "analysis")
        )
        
        # Setup fallback parser automatically
        self._setup_fallback_parser()
    
    def _setup_fallback_parser(self) -> None:
        """Setup fallback parser from configuration."""
        if "fallback_keywords" in self.prompt_config:
            parser = self._create_fallback_parser(self.prompt_config["fallback_keywords"])
            self.set_fallback_parser(parser)
    
    def _create_fallback_parser(self, keywords_config: Dict[str, List[str]]) -> Callable[[str], Dict[str, str]]:
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
    
    def set_fallback_parser(self, parser: Callable[[str], Dict[str, Any]]) -> None:
        """
        Set a custom fallback parser for non-JSON responses.
        Args:
            parser: Callable that takes a string and returns a parsed dict
        """
        self.fallback_parser = parser
    
    def load_and_process_images(self, image_folder: str) -> List[Dict[str, str]]:
        """
        Load, resize, and encode images from the specified folder.
        Args:
            image_folder: Path to the folder containing images
        Returns:
            List of dicts with 'name', 'data', and 'timestamp' for each image
        """
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
        """
        Create a standardized error result.
        Args:
            image_name: Name of the image
            error_message: Error message to include
        Returns:
            Dict with error information
        """
        return self.bedrock_client.create_analysis_result(
            image_name=image_name,
            parsed_response={"answer": "Error", "note": error_message},
            usage_info={},
            status="error"
        )
    
    async def analyze_image_async(self, image_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze a single image asynchronously with concurrency control.
        Args:
            image_data: Dict with image data (name, data, timestamp)
        Returns:
            Dict with analysis result or error result
        """
        async with self.semaphore:
            try:
                self.logger.info(f"Analyzing image: {image_data['name']}")
                
                # Send to Bedrock client using configured parameters
                response = self.bedrock_client.invoke_model(
                    prompt=self.prompt_config["main_prompt"],
                    max_tokens=self.model_params.get("max_tokens", 200),
                    temperature=self.model_params.get("temperature", 0.1),
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
    
    async def analyze_images_batch(self, encoded_images: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple images concurrently.
        Args:
            encoded_images: List of dicts with image data
        Returns:
            List of dicts with analysis results
        """
        self.logger.info(f"Starting batch analysis of {len(encoded_images)} images")
        
        # Create tasks for concurrent processing
        tasks = [
            self.analyze_image_async(image_data) 
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
    
    async def analyze_images(self, image_folder: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Analyze images using the configured analysis type.
        Args:
            image_folder: Optional folder path. If None, uses configured default.
        Returns:
            List of analysis results (one per image)
        """
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
        results = await self.analyze_images_batch(encoded_images)
        
        return results
    
    def format_results(self, results: List[Dict[str, Any]], analysis_type: Optional[str] = None) -> str:
        """
        Format analysis results as a readable string.
        Args:
            results: List of analysis result dicts
            analysis_type: Optional override for display type
        Returns:
            Formatted string for display
        """
        display_type = analysis_type or self.analysis_type.replace("_", " ").title()
        
        lines = []
        lines.append("\n" + "="*80)
        lines.append(f"{display_type.upper()} RESULTS")
        lines.append("="*80)
        
        total_cost = 0.0
        status_counts = {}
        
        for i, result in enumerate(results, 1):
            lines.append(f"\n{i}. Image: {result['image_name']}")
            lines.append(f"   Status: {result.get('status', 'Unknown')}")
            
            # Display main fields (answer, note, etc.)
            for key, value in result.items():
                if key not in ['image_name', 'status', 'input_tokens', 'output_tokens', 
                             'input_cost', 'output_cost']:
                    lines.append(f"   {key.title()}: {value}")
            
            lines.append(f"   Tokens: {result.get('input_tokens', 0)} input, {result.get('output_tokens', 0)} output")
            lines.append(f"   Cost: ${result.get('input_cost', 0) + result.get('output_cost', 0):.4f}")
            
            # Count statuses
            status = result.get('status', 'Unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            total_cost += result.get('input_cost', 0) + result.get('output_cost', 0)
        
        # Summary
        lines.append("\n" + "-"*80)
        lines.append("SUMMARY")
        lines.append("-"*80)
        lines.append(f"Total images analyzed: {len(results)}")
        for status, count in status_counts.items():
            lines.append(f"{status.title()}: {count}")
        lines.append(f"Total cost: ${total_cost:.4f}")
        lines.append("="*80)
        
        return "\n".join(lines)
    
    def display_results(self, results: List[Dict[str, Any]], analysis_type: Optional[str] = None) -> None:
        """
        Display analysis results in a readable format.
        Args:
            results: List of analysis result dicts
            analysis_type: Optional override for display type
        """
        formatted_results = self.format_results(results, analysis_type)
        print(formatted_results) 

    def analyze_image_grids(self, image_folder: str, gridder: 'ImageGridder', prompt_config: dict = None, model_params: dict = None) -> dict:
        """
        Analyze images in a folder as grids for multi-image LLM analysis.
        Args:
            image_folder: Path to the folder containing images
            gridder: An ImageGridder instance
            prompt_config: Optional prompt config dict (default: self.prompt_config)
            model_params: Optional model params dict (default: self.model_params)
        Returns:
            Dict with 'parsed_response' (parsed model output) and 'raw_response' (original model response)
            or dict with 'error' key if grid creation fails
        """
        from ..clients.bedrock_client import BedrockClient
        if prompt_config is None:
            prompt_config = self.prompt_config
        if model_params is None:
            model_params = self.model_params
        # Create grid images
        grids = gridder.create_grids(image_folder)
        if not grids:
            self.logger.error("No grid images could be created from input images.")
            return {"error": "No grid images created"}
        # Encode grid images
        encoded_grids = gridder.encode_grids(grids, format="PNG")
        # Multi-image analysis
        response = self.bedrock_client.invoke_model_multi_image(
            prompt=prompt_config["main_prompt"],
            images=encoded_grids,
            max_tokens=model_params.get("max_tokens", 2000),
            temperature=model_params.get("temperature", 0.1)
        )
        response_text = response.get("text", "")
        response_text = BedrockClient.repair_json_response(response_text)
        parsed_response = self.bedrock_client.parse_json_response(
            response_text,
            self.fallback_parser
        )
        return {"parsed_response": parsed_response, "raw_response": response} 