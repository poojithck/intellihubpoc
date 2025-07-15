from __future__ import annotations

import boto3
import json
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import base64

class Bedrock_Client:
    
    def __init__(self, model_id: str, region_name: Optional[str] = 'ap-southeast-2') -> None:
        """
        Initialize the Bedrock client.
        
        Args:
            model_id: The model ID to use (e.g., 'anthropic.claude-3-sonnet-20240229-v1:0')
            region_name: AWS region name (optional, will use default if not specified)
        """
        self.model_id = model_id
        self.region_name = region_name
        
        try:
            if region_name:
                self.client = boto3.client('bedrock-runtime', region_name=region_name)
            else:
                self.client = boto3.client('bedrock-runtime')
            logging.info(f"Bedrock client initialized successfully for model: {model_id}")
        except Exception as e:
            logging.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise
    
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
            
            # Ensure image data is base64 encoded
            if not self._is_base64(image_data):
                image_data = base64.b64encode(image_data.encode()).decode()
            
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
                            "text": prompt.strip()
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
        
        # Calculate costs (Claude 3.5 Sonnet pricing)
        input_price = 0.003 / 1000  # $0.003 per 1K input tokens
        output_price = 0.015 / 1000  # $0.015 per 1K output tokens
        
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
            # Try to decode as base64
            base64.b64decode(data)
            return True
        except Exception:
            return False
    
    