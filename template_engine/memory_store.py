"""
In-Memory Template Storage
Stores active templates received from Backend
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TemplateMemoryStore:
    """
    Singleton class to store templates in memory
    Templates are organized by category for faster lookup
    """
    
    _instance = None
    _templates: Dict[str, Dict] = {}  # {template_id: template_data}
    _templates_by_category: Dict[str, List[str]] = {}  # {category: [template_ids]}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._templates = {}
            cls._instance._templates_by_category = {}
            logger.info("TemplateMemoryStore initialized")
        return cls._instance
    
    def add_template(self, template: Dict) -> bool:
        """
        Add or update template in memory
        
        Args:
            template: Template dict with all configuration
            
        Returns:
            bool: Success status
        """
        try:
            template_id = template.get("template_id")
            category = template.get("category", "").lower()
            
            if not template_id:
                logger.error("Template missing template_id")
                return False
            
            # Store template
            self._templates[template_id] = template
            
            # Index by category
            if category:
                if category not in self._templates_by_category:
                    self._templates_by_category[category] = []
                
                if template_id not in self._templates_by_category[category]:
                    self._templates_by_category[category].append(template_id)
            
            logger.info(f"Template {template_id} added to memory (category: {category})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding template: {e}")
            return False
    
    def remove_template(self, template_id: str) -> bool:
        """
        Remove template from memory
        
        Args:
            template_id: Template ID to remove
            
        Returns:
            bool: Success status
        """
        try:
            if template_id not in self._templates:
                logger.warning(f"Template {template_id} not found in memory")
                return False
            
            # Get category before removing
            category = self._templates[template_id].get("category", "").lower()
            
            # Remove from main storage
            del self._templates[template_id]
            
            # Remove from category index
            if category and category in self._templates_by_category:
                if template_id in self._templates_by_category[category]:
                    self._templates_by_category[category].remove(template_id)
                    
                # Clean up empty category
                if not self._templates_by_category[category]:
                    del self._templates_by_category[category]
            
            logger.info(f"Template {template_id} removed from memory")
            return True
            
        except Exception as e:
            logger.error(f"Error removing template: {e}")
            return False
    
    def get_template(self, template_id: str) -> Optional[Dict]:
        """
        Get template by ID
        
        Args:
            template_id: Template ID
            
        Returns:
            Template dict or None
        """
        return self._templates.get(template_id)
    
    def get_templates_by_category(self, category: str) -> List[Dict]:
        """
        Get all templates for a category
        
        Args:
            category: Category name (BOL, Receipt, etc.)
            
        Returns:
            List of template dicts
        """
        category = category.lower()
        template_ids = self._templates_by_category.get(category, [])
        return [self._templates[tid] for tid in template_ids if tid in self._templates]
    
    def get_all_templates(self) -> List[Dict]:
        """Get all templates in memory"""
        return list(self._templates.values())
    
    def clear_all(self) -> None:
        """Clear all templates (use with caution)"""
        self._templates.clear()
        self._templates_by_category.clear()
        logger.warning("All templates cleared from memory")
    
    def get_stats(self) -> Dict:
        """Get memory store statistics"""
        return {
            "total_templates": len(self._templates),
            "categories": list(self._templates_by_category.keys()),
            "templates_by_category": {
                cat: len(tids) for cat, tids in self._templates_by_category.items()
            }
        }


# Singleton instance
_template_store = None


def get_template_store() -> TemplateMemoryStore:
    """Get singleton instance of TemplateMemoryStore"""
    global _template_store
    if _template_store is None:
        _template_store = TemplateMemoryStore()
    return _template_store