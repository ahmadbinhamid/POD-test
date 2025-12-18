"""
Prompt Loader
Loads prompts from template configuration
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Loads prompts from template configuration
    Falls back to default prompts if not found
    """
    
    def __init__(self, default_prompts: Optional[Dict] = None):
        """
        Initialize with optional default prompts
        
        Args:
            default_prompts: Dict of default prompts from prompts.py
        """
        self.default_prompts = default_prompts or {}
    
    def get_prompt(
        self,
        template: Dict,
        region_name: str,
        prompt_type: Optional[str] = None
    ) -> str:
        """
        Get prompt for a specific region
        
        Args:
            template: Template dict with prompts configuration
            region_name: Name of the region (e.g., "stamp", "bill_of_lading")
            prompt_type: Optional prompt type override
            
        Returns:
            Prompt string
        """
        # Get prompts from template
        prompts = template.get("prompts", {})
        
        # Try to get prompt for this region
        prompt = prompts.get(region_name)
        
        if prompt:
            logger.debug(f"Using template prompt for region: {region_name}")
            return prompt
        
        # Fallback to default prompts
        if region_name in self.default_prompts:
            logger.debug(f"Using default prompt for region: {region_name}")
            return self.default_prompts[region_name]
        
        # Generic fallback
        logger.warning(f"No prompt found for region: {region_name}, using generic prompt")
        return self._get_generic_prompt(region_name)
    
    def get_all_prompts(self, template: Dict) -> Dict[str, str]:
        """
        Get all prompts defined in template
        
        Args:
            template: Template dict
            
        Returns:
            Dict of {region_name: prompt}
        """
        return template.get("prompts", {})
    
    def _get_generic_prompt(self, region_name: str) -> str:
        """Generate generic prompt for a region"""
        return (
            f"Extract all relevant information from this {region_name} region. "
            f"Return the data in a structured JSON format with appropriate field names."
        )
    
    def build_batch_prompts(
        self,
        template: Dict,
        detected_regions: Dict[str, any]
    ) -> Dict[str, str]:
        """
        Build prompts for all detected regions
        
        Args:
            template: Template dict
            detected_regions: Dict of detected region names
            
        Returns:
            Dict of {region_name: prompt}
        """
        batch_prompts = {}
        
        for region_name in detected_regions.keys():
            prompt = self.get_prompt(template, region_name)
            batch_prompts[region_name] = prompt
        
        return batch_prompts