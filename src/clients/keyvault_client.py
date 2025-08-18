from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any
from azure.identity import DefaultAzureCredential
# TODO: Add missing SecretClient import for Key Vault integration
# from azure.keyvault.secrets import SecretClient
from azure.keyvault.secrets import SecretClient

class KeyVaultClient:
    """
    Azure Key Vault client for secure credential management.
    
    This is a placeholder implementation. You'll need to implement
    the SSO/Fabric authentication method based on your organization's setup.
    """
    
    def __init__(self, vault_url: str, credential: Optional[Any] = None):
        """
        Initialize the Key Vault client.
        
        Args:
            vault_url: The Key Vault URL
            credential: Azure credential (will use DefaultAzureCredential if None)
        """
        self.vault_url = vault_url
        self.credential = credential or DefaultAzureCredential()
        self.client = None
        
        try:
            self.client = SecretClient(vault_url=vault_url, credential=self.credential)
            logging.info(f"Key Vault client initialized successfully for: {vault_url}")
        except Exception as e:
            logging.warning(f"Failed to initialize Key Vault client: {str(e)}")
            logging.info("Will use environment variables as fallback")
            self.client = None
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """
        Get a secret from Key Vault.
        
        Args:
            secret_name: Name of the secret to retrieve
            
        Returns:
            The secret value, or None if not found
        """
        try:
            if self.client:
                secret = self.client.get_secret(secret_name)
                return secret.value
            else:
                logging.warning("Key Vault client not available, cannot retrieve secret")
                return None
        except Exception as e:
            logging.error(f"Failed to retrieve secret '{secret_name}': {str(e)}")
            return None
    
    def get_azure_openai_credentials(self) -> Dict[str, str]:
        """
        Get Azure OpenAI credentials from Key Vault.
        
        Returns:
            Dictionary with endpoint, api_key, and deployment_name
        """
        credentials = {}
        
        # Try to get from Key Vault first
        if self.client:
            try:
                api_key = self.get_secret("azure-openai-api-key")
                if api_key:
                    credentials["api_key"] = api_key
                    logging.info("Retrieved API key from Key Vault")
            except Exception as e:
                logging.warning(f"Failed to get API key from Key Vault: {str(e)}")
        
        # Fallback to environment variables
        if "api_key" not in credentials:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if api_key:
                credentials["api_key"] = api_key
                logging.info("Using API key from environment variable")
        
        # Get other credentials from environment variables
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if endpoint:
            credentials["endpoint"] = endpoint
            
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if deployment:
            credentials["deployment_name"] = deployment
        
        return credentials
    
    def is_available(self) -> bool:
        """
        Check if Key Vault is available.
        
        Returns:
            True if Key Vault client is working, False otherwise
        """
        return self.client is not None
    
    @classmethod
    def from_config(cls, config_manager) -> 'KeyVaultClient':
        """
        Create a Key Vault client from configuration.
        
        Args:
            config_manager: ConfigManager instance
            
        Returns:
            Configured KeyVaultClient instance
        """
        azure_config = config_manager.get_azure_config()
        vault_name = azure_config.get("keyvault", {}).get("name")
        
        if not vault_name:
            raise ValueError("Key Vault name not configured in azure_config.yaml")
        
        vault_url = f"https://{vault_name}.vault.azure.net/"
        
        # TODO: Implement your SSO/Fabric authentication here
        # TODO: Complete Key Vault authentication integration - currently using env vars as fallback
        # For now, using DefaultAzureCredential which may work in some environments
        credential = DefaultAzureCredential()
        
        return cls(vault_url=vault_url, credential=credential)
