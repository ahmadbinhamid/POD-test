"""
In-Memory Template Storage
Stores active templates received from Backend via /api/templates/sync
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger("template_engine.memory_store")


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
            logger.info("=" * 70)
            logger.info("🎯 TemplateMemoryStore initialized (Singleton)")
            logger.info("=" * 70)
        return cls._instance
    
    def add_template(self, template: Dict) -> bool:
        """
        Add or update template in memory
        
        Args:
            template: Template dict with all configuration
            
        Returns:
            bool: Success status
        """
        logger.info("\n" + "=" * 70)
        logger.info("📥 ADD_TEMPLATE: Starting template addition/update")
        logger.info("=" * 70)
        
        try:
            template_id = template.get("template_id")
            category = template.get("category", "").lower()
            template_name = template.get("template_name", "Unknown")
            
            logger.info(f"  Template ID: {template_id}")
            logger.info(f"  Template Name: {template_name}")
            logger.info(f"  Category: {category}")
            
            if not template_id:
                logger.error("  ❌ Template missing template_id!")
                return False
            
            # Check if updating existing template
            is_update = template_id in self._templates
            action = "UPDATE" if is_update else "ADD"
            
            logger.info(f"  Action: {action}")
            
            # Store template
            self._templates[template_id] = template
            logger.info(f"  ✅ Template stored in memory")
            
            # Index by category
            if category:
                if category not in self._templates_by_category:
                    self._templates_by_category[category] = []
                    logger.info(f"  📁 Created new category index: {category}")
                
                if template_id not in self._templates_by_category[category]:
                    self._templates_by_category[category].append(template_id)
                    logger.info(f"  🔗 Linked template to category: {category}")
            
            # Log template details
            regions = template.get("region_config", {}).get("yolo_config", {}).get("classes", [])
            logger.info(f"  Regions configured: {len(regions)}")
            
            prompts = template.get("prompts", {})
            logger.info(f"  Prompts configured: {list(prompts.keys())}")
            
            field_mappings = template.get("field_mapping", {})
            logger.info(f"  Field mappings: {len(field_mappings)}")
            
            logger.info("=" * 70)
            logger.info(f"✅ Template {template_id} successfully {action}ED to memory")
            logger.info("=" * 70 + "\n")
            
            return True
            
        except Exception as e:
            logger.error("=" * 70)
            logger.error(f"❌ Error adding template: {e}")
            logger.exception("Full traceback:")
            logger.error("=" * 70 + "\n")
            return False
    
    def remove_template(self, template_id: str) -> bool:
        """Remove template from memory"""
        logger.info("\n" + "=" * 70)
        logger.info(f"🗑️  REMOVE_TEMPLATE: Removing template {template_id}")
        logger.info("=" * 70)
        
        try:
            if template_id not in self._templates:
                logger.warning(f"  ⚠️  Template {template_id} not found in memory")
                return False
            
            category = self._templates[template_id].get("category", "").lower()
            logger.info(f"  Category: {category}")
            
            # Remove from main storage
            del self._templates[template_id]
            logger.info(f"  ✅ Removed from main storage")
            
            # Remove from category index
            if category and category in self._templates_by_category:
                if template_id in self._templates_by_category[category]:
                    self._templates_by_category[category].remove(template_id)
                    logger.info(f"  ✅ Removed from category index")
                    
                if not self._templates_by_category[category]:
                    del self._templates_by_category[category]
                    logger.info(f"  🧹 Cleaned up empty category: {category}")
            
            logger.info("=" * 70)
            logger.info(f"✅ Template {template_id} successfully removed")
            logger.info("=" * 70 + "\n")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error removing template: {e}")
            logger.exception("Full traceback:")
            return False
    
    def get_template(self, template_id: str) -> Optional[Dict]:
        """Get template by ID"""
        template = self._templates.get(template_id)
        if template:
            logger.debug(f"📖 Retrieved template: {template_id}")
        else:
            logger.debug(f"⚠️  Template not found: {template_id}")
        return template
    
    def get_templates_by_category(self, category: str) -> List[Dict]:
        """Get all templates for a category"""
        logger.info(f"\n🔍 GET_TEMPLATES_BY_CATEGORY: {category}")
        category = category.lower()
        template_ids = self._templates_by_category.get(category, [])
        templates = [self._templates[tid] for tid in template_ids if tid in self._templates]
        
        logger.info(f"  Found {len(templates)} template(s) in category '{category}'")
        for t in templates:
            logger.info(f"    - {t.get('template_id')}: {t.get('template_name')}")
        
        return templates
    
    def get_all_templates(self) -> List[Dict]:
        """Get all templates in memory"""
        return list(self._templates.values())
    
    def clear_all(self) -> None:
        """Clear all templates (use with caution)"""
        self._templates.clear()
        self._templates_by_category.clear()
        logger.warning("⚠️  ALL TEMPLATES CLEARED FROM MEMORY!")
    
    def get_stats(self) -> Dict:
        """Get memory store statistics"""
        stats = {
            "total_templates": len(self._templates),
            "by_category": {
                cat: len(tids) for cat, tids in self._templates_by_category.items()
            }
        }
        
        logger.info("\n📊 MEMORY STORE STATS:")
        logger.info(f"  Total templates: {stats['total_templates']}")
        logger.info(f"  Categories: {list(stats['by_category'].keys())}")
        for cat, count in stats['by_category'].items():
            logger.info(f"    - {cat}: {count} template(s)")
        
        return stats


# Singleton instance
_template_store = None


def get_template_store() -> TemplateMemoryStore:
    """Get singleton instance of TemplateMemoryStore"""
    global _template_store
    if _template_store is None:
        _template_store = TemplateMemoryStore()
    return _template_store