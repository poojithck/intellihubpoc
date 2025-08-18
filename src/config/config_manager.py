from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration loading and access for the application."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files. If None, uses package root/configs
        """
        if config_dir is None:
            # Resolve config directory relative to package root
            package_root = Path(__file__).parent.parent.parent
            self.config_dir = package_root / "configs"
        else:
            self.config_dir = Path(config_dir)
        
        self._configs: Dict[str, Dict[str, Any]] = {}
        
        # Ensure config directory exists
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")
    
    def load_config(self, config_name: str, subdirectory: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_name: Name of the config file (without .yaml extension)
            subdirectory: Optional subdirectory within configs/
            
        Returns:
            Dictionary containing configuration data
        """
        # Create cache key
        cache_key = f"{subdirectory}/{config_name}" if subdirectory else config_name
        
        # Return cached config if available
        if cache_key in self._configs:
            return self._configs[cache_key]
        
        # Determine file path
        if subdirectory:
            config_path = self.config_dir / subdirectory / f"{config_name}.yaml"
        else:
            config_path = self.config_dir / f"{config_name}.yaml"
        
        # Load configuration
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
                
            # Cache the configuration
            self._configs[cache_key] = config_data
            logger.info(f"Loaded configuration: {config_path}")
            
            return config_data
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading config {config_path}: {e}")
            raise
    
    def get_aws_config(self) -> Dict[str, Any]:
        """Get AWS and Bedrock configuration."""
        return self.load_config("aws_config")
    
    def get_azure_config(self) -> Dict[str, Any]:
        """Get Azure OpenAI configuration."""
        return self.load_config("azure_config")
    
    def get_azure_openai_config(self) -> Dict[str, Any]:
        """Get Azure OpenAI specific configuration."""
        azure_config = self.get_azure_config()
        return azure_config.get("azure_openai", {})
    
    def get_azure_pricing_config(self) -> Dict[str, Any]:
        """Get Azure OpenAI pricing configuration."""
        azure_config = self.get_azure_openai_config()
        deployment_name = azure_config.get("deployment_name", "gpt-4o-mini")
        pricing = azure_config.get("pricing", {})
        
        # Map deployment names to pricing keys (deployment names match pricing keys exactly)
        deployment_to_pricing = {
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini", 
            "gpt-35-turbo": "gpt-35-turbo",
            "o3": "o3"
        }
        
        pricing_key = deployment_to_pricing.get(deployment_name, "gpt_4o_mini")
        
        if pricing_key in pricing:
            return pricing[pricing_key]
        else:
            # Default to gpt-4o-mini pricing
            return pricing.get("gpt_4o_mini", {
                "input_price_per_1k": 0.0001,
                "output_price_per_1k": 0.0002
            })
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration."""
        return self.load_config("app_config")
    
    def get_model_config(self, model_name: str = "claude_3_5_sonnet") -> Dict[str, Any]:
        """
        Get model-specific configuration.
        
        Args:
            model_name: Name of the model to get config for
            
        Returns:
            Model configuration dictionary
        """
        claude_config = self.load_config("claude_config", "models")
        
        if model_name not in claude_config.get("models", {}):
            raise ValueError(f"Model configuration not found: {model_name}")
        
        return claude_config["models"][model_name]
    
    def get_azure_model_config(self, model_name: str = "gpt_4o") -> Dict[str, Any]:
        """
        Get Azure OpenAI model-specific configuration.
        
        Args:
            model_name: Name of the model to get config for
            
        Returns:
            Model configuration dictionary
        """
        azure_config = self.get_azure_openai_config()
        models_config_file = azure_config.get("models_config_file", "configs/models/azure_openai_config.yaml")
        
        # Extract the filename from the path
        config_name = models_config_file.split("/")[-1].replace(".yaml", "")
        azure_models_config = self.load_config(config_name, "models")
        
        if model_name not in azure_models_config.get("models", {}):
            raise ValueError(f"Azure OpenAI model configuration not found: {model_name}")
        
        return azure_models_config["models"][model_name]
    
    def get_prompt_config(self, analysis_type: str, prompts_subdir: Optional[str] = None) -> Dict[str, Any]:
        """
        Get prompt configuration for a specific analysis type.
        
        Args:
            analysis_type: Type of analysis (e.g., "fuse_analysis")
            prompts_subdir: Optional subdirectory under configs/ to load prompts from.
                             Defaults to "prompts". Can be a nested path like
                             "prompts/Targeted-Prompts".
            
        Returns:
            Prompt configuration dictionary
        """
        subdir = prompts_subdir or "prompts"
        try:
            config_data = self.load_config(analysis_type, subdir)
        except FileNotFoundError:
            # Fallback to default prompts directory if targeted/nested not found
            if subdir != "prompts":
                config_data = self.load_config(analysis_type, "prompts")
            else:
                raise
        
        if analysis_type not in config_data:
            # If structure differs, attempt fallback once more to default
            if subdir != "prompts":
                fallback_data = self.load_config(analysis_type, "prompts")
                if analysis_type in fallback_data:
                    return fallback_data[analysis_type]
            raise ValueError(f"Prompt configuration not found: {analysis_type}")
        
        return config_data[analysis_type]
    
    def get_bedrock_client_config(self) -> Dict[str, Any]:
        """Get configuration for initializing Bedrock client."""
        aws_config = self.get_aws_config()
        
        return {
            "region_name": aws_config["aws"]["region"],
            "model_id": aws_config["bedrock"]["default_model"],
            "timeout": aws_config["bedrock"]["timeout"],
            "retry_attempts": aws_config["bedrock"]["retry_attempts"]
        }
    
    def get_model_params(self, model_name: str = "claude_3_5_sonnet", 
                        config_type: str = "default_params") -> Dict[str, Any]:
        """
        Get model parameters for a specific configuration type.
        
        Args:
            model_name: Name of the model
            config_type: Type of configuration (default_params, analysis, creative, etc.)
            
        Returns:
            Model parameters dictionary
        """
        model_config = self.get_model_config(model_name)
        
        if config_type == "default_params":
            return model_config.get("default_params", {})
        else:
            specialized_configs = model_config.get("specialized_configs", {})
            if config_type not in specialized_configs:
                logger.warning(f"Specialized config '{config_type}' not found, using default_params")
                return model_config.get("default_params", {})
            return specialized_configs[config_type]
    
    def get_azure_model_params(self, model_name: str = "gpt_4o", 
                              config_type: str = "default_params") -> Dict[str, Any]:
        """
        Get Azure OpenAI model parameters for a specific configuration type.
        
        Args:
            model_name: Name of the model
            config_type: Type of configuration (default_params, analysis, creative, etc.)
            
        Returns:
            Model parameters dictionary
        """
        model_config = self.get_azure_model_config(model_name)
        
        if config_type == "default_params":
            return model_config.get("default_params", {})
        else:
            specialized_configs = model_config.get("specialized_configs", {})
            if config_type not in specialized_configs:
                logger.warning(f"Specialized config '{config_type}' not found, using default_params")
                return model_config.get("default_params", {})
            return specialized_configs[config_type]
    
    def get_pricing_config(self, model_name: str = "claude_3_5_sonnet") -> Dict[str, Any]:
        """
        Get pricing configuration for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Pricing configuration dictionary
        """
        aws_config = self.get_aws_config()
        pricing = aws_config.get("pricing", {})
        
        if model_name not in pricing:
            logger.warning(f"Pricing config not found for {model_name}, using defaults")
            return {"input_price_per_1k": 0.003, "output_price_per_1k": 0.015}
        
        return pricing[model_name]
    
    def get_image_processing_config(self) -> Dict[str, Any]:
        """Get image processing configuration."""
        app_config = self.get_app_config()
        return app_config.get("image_processing", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        app_config = self.get_app_config()
        return app_config.get("logging", {})
    
    def reload_config(self, config_name: str, subdirectory: Optional[str] = None) -> None:
        """
        Reload a specific configuration file.
        
        Args:
            config_name: Name of the config file to reload
            subdirectory: Optional subdirectory within configs/
        """
        cache_key = f"{subdirectory}/{config_name}" if subdirectory else config_name
        
        # Remove from cache to force reload
        if cache_key in self._configs:
            del self._configs[cache_key]
        
        # Reload the configuration
        self.load_config(config_name, subdirectory)
        logger.info(f"Reloaded configuration: {cache_key}")
    
    def reload_all_configs(self) -> None:
        """Reload all cached configurations."""
        self._configs.clear()
        logger.info("Cleared all cached configurations")
    
    def list_available_configs(self) -> Dict[str, list]:
        """List all available configuration files."""
        configs = {"root": [], "subdirectories": {}}
        
        # Root level configs
        for config_file in self.config_dir.glob("*.yaml"):
            configs["root"].append(config_file.stem)
        
        # Subdirectory configs
        for subdir in self.config_dir.iterdir():
            if subdir.is_dir():
                subdir_configs = []
                for config_file in subdir.glob("*.yaml"):
                    subdir_configs.append(config_file.stem)
                if subdir_configs:
                    configs["subdirectories"][subdir.name] = subdir_configs
        
        return configs
    
    def get_config(self, config_name: str, subdirectory: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration data (alias for load_config for consistency).
        
        Args:
            config_name: Name of the config file (without .yaml extension)
            subdirectory: Optional subdirectory within configs/
            
        Returns:
            Dictionary containing configuration data
        """
        return self.load_config(config_name, subdirectory) 