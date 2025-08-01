from __future__ import annotations

import boto3
import json
from typing import Optional, Dict, Any, List, Tuple
import logging
import base64
import binascii

class BedrockClient:
    
    def __init__(self, model_id: str, region_name: Optional[str] = 'ap-southeast-2',
                 pricing_config: Optional[Dict[str, float]] = None) -> None:
        """
        Initialize the Bedrock client.
        
        Args:
            model_id: The model ID to use (e.g., 'anthropic.claude-3-sonnet-20240229-v1:0')
            region_name: AWS region name (optional, will use default if not specified)
            pricing_config: Optional pricing configuration for cost calculation
        """
        self.model_id = model_id
        self.region_name = region_name
        
        # Set pricing configuration
        self.pricing_config = pricing_config or {
            "input_price_per_1k": 0.003,
            "output_price_per_1k": 0.015
        }
        
        try:
            if region_name:
                self.client = boto3.client('bedrock-runtime', region_name=region_name)
            else:
                self.client = boto3.client('bedrock-runtime')
            logging.info(f"Bedrock client initialized successfully for model: {model_id}")
        except Exception as e:
            logging.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise
    
    @classmethod
    def from_config(cls, config_manager) -> 'BedrockClient':
        """
        Create a Bedrock client from configuration.
        
        Args:
            config_manager: ConfigManager instance
            
        Returns:
            Configured BedrockClient instance
        """
        bedrock_config = config_manager.get_bedrock_client_config()
        pricing_config = config_manager.get_pricing_config()
        
        return cls(
            model_id=bedrock_config["model_id"],
            region_name=bedrock_config["region_name"],
            pricing_config=pricing_config
        )
    
    def invoke_model(
        self, 
        prompt: str, 
        max_tokens: int = 300, 
        temperature: float = 0.7,
        images: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Invoke the model with the given prompt and optional images.
        
        Args:
            prompt: The text prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0 = deterministic, 1.0 = very random)
            images: Optional list of image dictionaries with 'name' and 'data' (base64) keys
            
        Returns:
            Dictionary containing the model response and metadata
        """
        try:
            if images:
                return self._invoke_multimodal_model(prompt, max_tokens, temperature, images)
            else:
                return self._invoke_text_model(prompt, max_tokens, temperature)
        except Exception as e:
            logging.error(f"Failed to invoke model: {str(e)}")
            raise
    
    def invoke_model_multi_image(
        self, 
        prompt: str, 
        images: List[Dict[str, str]],
        max_tokens: int = 1000, 
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Invoke model with multiple images in a single request for comparison analysis.
        
        Args:
            prompt: The text prompt to send to the model
            images: List of image dictionaries with 'name', 'data' (base64), and optional 'timestamp' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0 = deterministic, 1.0 = very random)
            
        Returns:
            Dictionary containing the model response and metadata
        """
        if not images:
            raise ValueError("At least one image must be provided")
        
        # Build content array with all images and text
        content = []
        
        # Add text prompt first
        content.append({
            "type": "text",
            "text": prompt.strip()
        })
        
        # Add each image with optional timestamp info
        for i, image_info in enumerate(images):
            name = image_info['name']
            image_data = image_info['data']
            timestamp = image_info.get('timestamp')
            
            # Ensure image data is base64 encoded
            if not self._is_base64(image_data):
                image_data = base64.b64encode(image_data.encode()).decode()
            
            # Add image
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data
                }
            })
            
            # Add image metadata as text
            metadata_text = f"Image {i+1:02d} of {len(images):02d}: {name}"
            if timestamp:
                metadata_text += f" (captured: {timestamp})"
            
            content.append({
                "type": "text", 
                "text": metadata_text
            })
        
        # Build request
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body),
            contentType='application/json',
            accept='application/json'
        )
        
        return self._parse_response(response)
    
    def _invoke_text_model(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Invoke model with text-only prompt."""
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt.strip()
                }
            ]
        }
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body),
            contentType='application/json',
            accept='application/json'
        )
        
        return self._parse_response(response)
    
    def _invoke_multimodal_model(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float, 
        images: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Invoke model with text and image prompts."""
        payloads = self._generate_multimodal_payloads(prompt, max_tokens, temperature, images)
        responses = self._call_model_batch(payloads)
        return self._parse_multimodal_responses(responses)
    
    def _generate_multimodal_payloads(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float, 
        images: List[Dict[str, str]]
    ) -> List[Tuple[str, str]]:
        """Generate payloads for multimodal requests."""
        payloads = []
        
        for image_info in images:
            name = image_info['name']
            image_data = image_info['data']
            timestamp = image_info.get('timestamp')
            
            # Ensure image data is base64 encoded
            if not self._is_base64(image_data):
                image_data = base64.b64encode(image_data.encode()).decode()
            
            # Build prompt, optionally prepending timestamp information so the model can reason about temporal order
            if timestamp:
                prompt_text = f"Image capture timestamp: {timestamp}. {prompt.strip()}"
            else:
                prompt_text = prompt.strip()

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ]
            
            request = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
            
            payloads.append((name, json.dumps(request)))
        
        return payloads
    
    def _call_model_batch(self, payloads: List[Tuple[str, str]]) -> List[Tuple[str, Any]]:
        """Call the model for a batch of payloads."""
        response_list = []
        
        for name, payload in payloads:
            logging.info(f"Calling model for image: {name}")
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=payload,
                contentType='application/json',
                accept='application/json'
            )
            
            logging.info(f"Response received for image: {name}")
            response_list.append((name, response))
        
        return response_list
    
    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse a single model response."""
        response_body = json.loads(response['body'].read())
        
        # Extract usage information
        usage = response_body.get("usage", {})
        input_tokens = usage.get("inputTokens") or usage.get("input_tokens", 0)
        output_tokens = usage.get("outputTokens") or usage.get("output_tokens", 0)
        total_tokens = usage.get("totalTokens") or usage.get("total_tokens", 0)
        
        # Extract generated text
        generated_text = response_body['content'][0]['text']
        
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
    
    def _parse_multimodal_responses(self, response_list: List[Tuple[str, Any]]) -> Dict[str, Any]:
        """Parse multiple model responses from multimodal requests."""
        results = []
        total_cost = 0.0
        
        for name, response in response_list:
            parsed_response = self._parse_response(response)
            
            # Try to parse the generated text as JSON if it looks like JSON
            try:
                if parsed_response["text"].strip().startswith("{"):
                    response_dict = json.loads(parsed_response["text"])
                else:
                    response_dict = {"raw_text": parsed_response["text"]}
            except json.JSONDecodeError:
                response_dict = {"raw_text": parsed_response["text"]}
            
            # Add metadata
            response_dict.update({
                "input_tokens": parsed_response["input_tokens"],
                "output_tokens": parsed_response["output_tokens"],
                "input_cost": parsed_response["input_cost"],
                "output_cost": parsed_response["output_cost"],
                "image_name": name
            })
            
            results.append(response_dict)
            total_cost += parsed_response["total_cost"]
        
        logging.info(f"Total cost of batch: ${total_cost:.4f}")
        
        return {
            "results": results,
            "total_cost": total_cost,
            "count": len(results)
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
    
    