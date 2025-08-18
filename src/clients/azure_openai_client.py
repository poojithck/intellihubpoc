from __future__ import annotations

import json
from typing import Optional, Dict, Any, List, Tuple
import logging
import base64
import binascii
from datetime import datetime
from openai import AzureOpenAI

class AzureOpenAIClient:
    
    def __init__(self, endpoint: str, deployment_name: str, api_key: str, api_version: str = "2024-02-15-preview",
                 pricing_config: Optional[Dict[str, float]] = None) -> None:
        """
        Initialize the Azure OpenAI client.
        
        Args:
            endpoint: The Azure OpenAI endpoint URL
            deployment_name: The deployment name to use
            api_key: The API key for authentication
            api_version: API version for Azure OpenAI
            pricing_config: Optional pricing configuration for cost calculation
        """
        self.endpoint = endpoint.rstrip('/')
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.api_version = api_version
        
        # Set pricing configuration (Azure OpenAI pricing)
        self.pricing_config = pricing_config or {
            "input_price_per_1k": 0.0001,  # GPT-4o-mini input tokens
            "output_price_per_1k": 0.0002   # GPT-4o-mini output tokens
        }
        
        # Initialize the official Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        logging.info(f"Azure OpenAI client initialized successfully for deployment: {deployment_name}")
    
    @classmethod
    def from_config(cls, config_manager) -> 'AzureOpenAIClient':
        """
        Create an Azure OpenAI client from configuration.
        
        Args:
            config_manager: ConfigManager instance
            
        Returns:
            Configured AzureOpenAIClient instance
        """
        azure_config = config_manager.get_azure_openai_config()
        pricing_config = config_manager.get_azure_pricing_config()
        
        # Get API key from environment variable or config
        # TODO: Integrate Key Vault authentication once keyvault_client.py is fixed
        # Currently using environment variables as fallback
        import os
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable must be set")
        
        return cls(
            endpoint=azure_config["endpoint"],
            deployment_name=azure_config["deployment_name"],
            api_key=api_key,
            api_version=azure_config.get("api_version", "2024-02-15-preview"),
            pricing_config=pricing_config
        )
    
    def invoke_model(
        self, 
        prompt: str, 
        max_tokens: int = 300, 
        temperature: float = 0.7,
        images: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Invoke the model with the given prompt and optional images.
        
        Args:
            prompt: The text prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0 = deterministic, 1.0 = very random)
            images: Optional list of image dictionaries with 'name' and 'data' (base64) keys
            system_prompt: Optional system prompt to align with Bedrock approach
            
        Returns:
            Dictionary containing the model response and metadata
        """
        try:
            if images:
                return self._invoke_multimodal_model(prompt, max_tokens, temperature, images, system_prompt)
            else:
                return self._invoke_text_model(prompt, max_tokens, temperature, system_prompt)
        except Exception as e:
            logging.error(f"Failed to invoke model: {str(e)}")
            raise
    
    def invoke_model_multi_image(
        self, 
        prompt: str, 
        images: List[Dict[str, str]],
        max_tokens: int = 1000, 
        temperature: float = 0.1,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Invoke model with multiple images in a single request for comparison analysis.
        
        Args:
            prompt: The main user prompt to send to the model
            images: List of image dictionaries with 'name', 'data' (base64), and optional 'timestamp' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0 = deterministic, 1.0 = very random)
            system_prompt: Optional system prompt to align with Bedrock approach
            
        Returns:
            Dictionary containing the model response and metadata
        """
        if not images:
            raise ValueError("At least one image must be provided")
        
        # Enhanced debug logging for accuracy verification
        logging.info(f"Azure API Debug - Multi-image request")
        logging.info(f"   Prompt: {len(prompt)} chars")
        logging.info(f"   Images: {len(images)}")
        logging.info(f"   Max tokens: {max_tokens}, Temperature: {temperature}")
        if system_prompt:
            logging.info(f"   System prompt: {len(system_prompt)} chars")
        
        # Build content array with all images and text
        content = []
        
        # Integrate timestamp into prompt text like Bedrock approach
        prompt_text = prompt.strip()
        if len(images) == 1 and images[0].get('timestamp'):
            # For single images, integrate timestamp into prompt like Bedrock
            timestamp = images[0]['timestamp']
            prompt_text = f"Image capture timestamp: {timestamp}. {prompt_text}"
            logging.info(f"   Integrated timestamp: {timestamp}")
        
        # Add text prompt first
        content.append({
            "type": "text",
            "text": prompt_text
        })
        
        # Add each image with optional timestamp info
        for i, image_info in enumerate(images):
            name = image_info['name']
            image_data = image_info['data']
            timestamp = image_info.get('timestamp')
            
            # Ensure image data is base64 encoded
            if not self._is_base64(image_data):
                image_data = base64.b64encode(image_data.encode()).decode()
            
            # Check image data size before sending to Azure
            try:
                encoded_size = len(image_data.encode('utf-8'))
                encoded_mb = encoded_size / (1024 * 1024)
                logging.info(f"   Image {i+1}: {name} - {encoded_mb:.2f}MB")
                
                if encoded_size > 20 * 1024 * 1024:  # Azure OpenAI 20MB limit
                    logging.error(f"IMAGE SIZE LIMIT EXCEEDED!")
                    logging.error(f"OVERSIZED IMAGE DETECTED: {encoded_mb:.2f}MB ({encoded_size} bytes)")
                    logging.error(f"Image name: {name}")
                    raise ValueError(f"BLOCKED OVERSIZED IMAGE: {encoded_mb:.2f}MB ({encoded_size} bytes) from {name} - SIZE LIMIT EXCEEDED!")
            except ValueError:
                raise
            except Exception as size_check_error:
                logging.error(f"Failed to validate image size for {name}: {size_check_error}")
                raise
            
            # Add image
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}"
                }
            })
            
            # Add image metadata as text (only for multi-image scenarios)
            if len(images) > 1:
                metadata_text = f"Image {i+1:02d} of {len(images):02d}: {name}"
                if timestamp:
                    metadata_text += f" (captured: {timestamp})"
                content.append({
                    "type": "text", 
                    "text": metadata_text
                })
        
        # Build messages array with system message support like Bedrock
        messages = []
        
        # Add system message if provided (aligning with Bedrock approach)
        if system_prompt and system_prompt.strip():
            messages.append({
                "role": "system",
                "content": system_prompt.strip()
            })
            logging.info(f"   Added system message: {len(system_prompt.strip())} chars")
        
        # Add user message with content
        messages.append({
            "role": "user",
            "content": content
        })
        
        # Log final message structure for verification
        logging.info(f"   Final message structure: {len(messages)} messages, {len(content)} content items")
        
        # Make the API call using the official client
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return self._parse_azure_response(response)
    
    def invoke_model_per_image(
        self,
        prompt: str,
        images: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Invoke model with each image individually (Bedrock-style approach for accuracy).
        Processes each image separately and combines results.
        
        Args:
            prompt: The main user prompt to send to the model
            images: List of image dictionaries with 'name', 'data' (base64), and optional 'timestamp' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0 = deterministic, 1.0 = very random)
            system_prompt: Optional system prompt to align with Bedrock approach
            
        Returns:
            Dictionary containing combined model response and metadata
        """
        if not images:
            raise ValueError("At least one image must be provided")
        
        # Enhanced debug logging for per-image processing
        logging.info(f"Azure API Debug - Per-image processing (Bedrock-style)")
        logging.info(f"   Base prompt: {len(prompt)} chars")
        logging.info(f"   Total images: {len(images)}")
        logging.info(f"   Max tokens: {max_tokens}, Temperature: {temperature}")
        if system_prompt:
            logging.info(f"   System prompt: {len(system_prompt)} chars")
        
        results = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        
        for i, image_info in enumerate(images):
            name = image_info['name']
            timestamp = image_info.get('timestamp')
            
            logging.info(f"   Processing image {i+1}/{len(images)}: {name}")
            
            # Adjust prompt with timestamp like Bedrock does
            image_prompt = prompt
            if timestamp:
                image_prompt = f"Image capture timestamp: {timestamp}. {prompt.strip()}"
                logging.info(f"   Image {i+1} timestamp: {timestamp}")
            
            # Process single image
            try:
                logging.info(f"   Sending image {i+1} to Azure API...")
                response = self.invoke_model_multi_image(
                    prompt=image_prompt,
                    images=[image_info],  # Single image
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt
                )
                
                # Log individual response details
                img_response_text = response.get("text", "")
                img_input_tokens = response.get("input_tokens", 0)
                img_output_tokens = response.get("output_tokens", 0)
                img_cost = response.get("total_cost", 0.0)
                
                logging.info(f"   Image {i+1} response: {len(img_response_text)} chars, {img_input_tokens} input, {img_output_tokens} output, ${img_cost:.4f}")
                
                # Accumulate metrics
                total_input_tokens += response.get("input_tokens", 0)
                total_output_tokens += response.get("output_tokens", 0)
                total_cost += response.get("total_cost", 0.0)
                
                # Store individual result
                results.append({
                    "image_name": name,
                    "response": response.get("text", ""),
                    "timestamp": timestamp
                })
                
            except Exception as e:
                logging.error(f"Failed to process image {name}: {str(e)}")
                results.append({
                    "image_name": name,
                    "response": f"Error processing image: {str(e)}",
                    "timestamp": timestamp
                })
        
        # Combine all individual responses
        combined_text = "\n\n".join([
            f"Image: {result['image_name']}\nResponse: {result['response']}"
            for result in results
        ])
        
        logging.info(f"   Combined response: {len(combined_text)} chars, total cost: ${total_cost:.4f}")
        
        return {
            "text": combined_text,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "input_cost": total_input_tokens * (self.pricing_config["input_price_per_1k"] / 1000),
            "output_cost": total_output_tokens * (self.pricing_config["output_price_per_1k"] / 1000),
            "total_cost": total_cost,
            "individual_results": results,
            "processing_mode": "per_image"
        }
    
    def _invoke_text_model(self, prompt: str, max_tokens: int, temperature: float, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Invoke model with text-only prompt."""
        # Build messages array with system message support like Bedrock
        messages = []
        
        # Add system message if provided (aligning with Bedrock approach)
        if system_prompt and system_prompt.strip():
            messages.append({
                "role": "system",
                "content": system_prompt.strip()
            })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": prompt.strip()
        })
        
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return self._parse_azure_response(response)
    
    def _invoke_multimodal_model(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float, 
        images: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Invoke model with text and image prompts."""
        # For Azure OpenAI, we can send multiple images in a single request
        return self.invoke_model_multi_image(prompt, images, max_tokens, temperature, system_prompt)
    
    def _parse_azure_response(self, response) -> Dict[str, Any]:
        """Parse Azure OpenAI response to match BedrockClient format."""
        # Extract usage information
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        
        # Extract generated text
        generated_text = response.choices[0].message.content
        
        # Calculate costs using configured pricing
        input_price = self.pricing_config["input_price_per_1k"] / 1000
        output_price = self.pricing_config["output_price_per_1k"] / 1000
        
        input_cost = input_tokens * input_price
        output_cost = output_tokens * output_price
        total_cost = input_cost + output_cost
        
        return {
            "text": generated_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    
    def _is_base64(self, data: str) -> bool:
        """Check if a string is base64 encoded."""
        try:
            # Try to decode as base64 with validation
            base64.b64decode(data, validate=True)
            return True
        except (binascii.Error, ValueError):
            return False
    
    def parse_json_response(self, response_text: str, fallback_parser=None) -> Dict[str, Any]:
        """
        Parse JSON response from model output, with fallback handling.
        
        Args:
            response_text: The raw text response from the model
            fallback_parser: Optional function to parse non-JSON responses
            
        Returns:
            Dictionary containing parsed response
        """
        try:
            # Try to extract JSON from the raw text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # No JSON found, use fallback parser if provided
                if fallback_parser:
                    return fallback_parser(response_text)
                else:
                    return {"raw_text": response_text}
                    
        except json.JSONDecodeError:
            # JSON parsing failed, use fallback parser if provided
            if fallback_parser:
                return fallback_parser(response_text)
            else:
                return {"raw_text": response_text}
    
    @staticmethod
    def repair_json_response(response_text: str) -> str:
        """
        Attempt to repair truncated or malformed JSON in a model response.
        Args:
            response_text: The raw text response from the model
        Returns:
            A string containing the repaired JSON, or the original text if repair is not possible
        """
        json_start = response_text.find('{')
        if json_start != -1:
            json_text = response_text[json_start:]
            
            # Handle missing closing braces
            if json_text.count('{') > json_text.count('}'):
                last_comma = json_text.rfind(',')
                if last_comma > 0:
                    json_text = json_text[:last_comma] + '\n}'
                else:
                    json_text += '}'
            
            # Try to parse and validate the JSON
            try:
                parsed = json.loads(json_text)
                # If successful, return the cleaned JSON
                return json.dumps(parsed, separators=(',', ':'))
            except json.JSONDecodeError:
                # If parsing fails, try to extract the first valid JSON object
                try:
                    # Find the first complete JSON object
                    brace_count = 0
                    end_pos = -1
                    for i, char in enumerate(json_text):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    
                    if end_pos > 0:
                        valid_json = json_text[:end_pos]
                        # Validate by parsing
                        json.loads(valid_json)
                        return valid_json
                except (json.JSONDecodeError, ValueError):
                    pass
                
                # If all else fails, return the original text
                return json_text
        return response_text
    
    def create_analysis_result(self, image_name: str, parsed_response: Dict[str, Any], 
                             usage_info: Dict[str, Any], status: str = "success") -> Dict[str, Any]:
        """
        Create a standardized analysis result structure.
        
        Args:
            image_name: Name of the analyzed image
            parsed_response: Parsed response from the model
            usage_info: Token usage and cost information
            status: Status of the analysis (success, error, etc.)
            
        Returns:
            Standardized result dictionary
        """
        base_result = {
            "image_name": image_name,
            "status": status,
            "input_tokens": usage_info.get("input_tokens", 0),
            "output_tokens": usage_info.get("output_tokens", 0),
            "input_cost": usage_info.get("input_cost", 0),
            "output_cost": usage_info.get("output_cost", 0)
        }
        
        # Merge with parsed response
        base_result.update(parsed_response)
        return base_result
    
    @staticmethod
    def create_fallback_parser(sor_type: str):
        """Create a simple fallback parser for a specific SOR type."""
        def fallback_parser(text: str) -> Dict[str, Any]:
            """Simple fallback parser that extracts basic fields from text."""
            result = {}
            
            # Try to extract common boolean fields
            boolean_fields = {
                "valid_claim": ["valid", "claim", "true", "pass", "yes"],
                "valid_installation": ["valid", "installation", "installed", "true", "pass", "yes"],
                "valid_consolidation": ["valid", "consolidation", "consolidated", "true", "pass", "yes"],
                "valid_removal": ["valid", "removal", "removed", "true", "pass", "yes"],
                "devices_added": ["device", "added", "installed", "true", "pass", "yes"],
                "meters_removed": ["meter", "removed", "true", "pass", "yes"],
                "switch_installed": ["switch", "installed", "true", "pass", "yes"],
                "neutral_link_installed": ["neutral", "link", "installed", "true", "pass", "yes"]
            }
            
            text_lower = text.lower()
            for field, keywords in boolean_fields.items():
                result[field] = any(keyword in text_lower for keyword in keywords)
            
            # Extract notes if possible
            result["notes"] = f"Fallback parsing used for {sor_type}"
            
            return result
        
        return fallback_parser
