"""
Post-Processor
Applies template-specific post-processing rules
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Applies post-processing rules defined in template
    - Date normalization
    - Quantity validation
    - Conditional logic
    - Field transformations
    """
    
    def apply_rules(
        self,
        extracted_data: Dict[str, Any],
        template: Dict
    ) -> Dict[str, Any]:
        """
        Apply all post-processing rules from template
        
        Args:
            extracted_data: Data extracted from VLM
            template: Template dict with post_processing_rules
            
        Returns:
            Processed data dict
        """
        rules = template.get("post_processing_rules", [])
        
        if not rules:
            logger.debug("No post-processing rules defined in template")
            return extracted_data
        
        processed_data = extracted_data.copy()
        
        for rule in rules:
            try:
                processed_data = self._apply_single_rule(processed_data, rule)
            except Exception as e:
                logger.error(f"Error applying rule {rule.get('rule_id')}: {e}")
        
        return processed_data
    
    def _apply_single_rule(
        self,
        data: Dict[str, Any],
        rule: Dict
    ) -> Dict[str, Any]:
        """Apply a single post-processing rule"""
        rule_type = rule.get("rule_type", "").lower()
        
        if rule_type == "date_normalization":
            return self._apply_date_normalization(data, rule)
        elif rule_type == "quantity_validation":
            return self._apply_quantity_validation(data, rule)
        elif rule_type == "conditional_logic":
            return self._apply_conditional_logic(data, rule)
        elif rule_type == "field_transformation":
            return self._apply_field_transformation(data, rule)
        elif rule_type == "notation_logic":
            return self._apply_notation_logic(data, rule)
        else:
            logger.warning(f"Unknown rule type: {rule_type}")
            return data
    
    def _apply_date_normalization(
        self,
        data: Dict[str, Any],
        rule: Dict
    ) -> Dict[str, Any]:
        """Normalize date fields to specified format"""
        target_field = rule.get("target_field")
        desired_format = rule.get("desired_format", "%m/%d/%y")
        
        if not target_field or target_field not in data:
            return data
        
        date_value = data[target_field]
        
        if not date_value or date_value == "null":
            return data
        
        try:
            # Try to parse various date formats
            normalized_date = self._parse_and_format_date(str(date_value), desired_format)
            data[target_field] = normalized_date
            logger.debug(f"Normalized {target_field}: {date_value} → {normalized_date}")
        except Exception as e:
            logger.warning(f"Could not normalize date {date_value}: {e}")
        
        return data
    
    def _apply_quantity_validation(
        self,
        data: Dict[str, Any],
        rule: Dict
    ) -> Dict[str, Any]:
        """Validate and correct quantity fields"""
        target_fields = rule.get("target_fields", [])
        
        for field in target_fields:
            if field in data:
                value = data[field]
                
                # Convert to number
                try:
                    num_value = float(value) if value not in [None, "null", ""] else 0
                    
                    # Validate non-negative
                    if num_value < 0:
                        logger.warning(f"Negative quantity in {field}: {num_value}, setting to 0")
                        data[field] = 0
                    else:
                        data[field] = int(num_value) if num_value.is_integer() else num_value
                        
                except (ValueError, TypeError, AttributeError):
                    logger.warning(f"Invalid quantity in {field}: {value}, setting to 0")
                    data[field] = 0
        
        return data
    
    def _apply_conditional_logic(
        self,
        data: Dict[str, Any],
        rule: Dict
    ) -> Dict[str, Any]:
        """Apply conditional logic rules"""
        condition_field = rule.get("condition_field")
        condition_value = rule.get("condition_value")
        actions = rule.get("actions", [])
        
        if not condition_field or condition_field not in data:
            return data
        
        # Check condition
        if str(data[condition_field]).lower() == str(condition_value).lower():
            # Apply actions
            for action in actions:
                target_field = action.get("target_field")
                new_value = action.get("value")
                
                if target_field:
                    data[target_field] = new_value
                    logger.debug(
                        f"Conditional rule applied: {condition_field}={condition_value} "
                        f"→ {target_field}={new_value}"
                    )
        
        return data
    
    def _apply_field_transformation(
        self,
        data: Dict[str, Any],
        rule: Dict
    ) -> Dict[str, Any]:
        """Transform field values using specified transformations"""
        target_field = rule.get("target_field")
        transformation = rule.get("transformation", "").lower()
        
        if not target_field or target_field not in data:
            return data
        
        value = data[target_field]
        
        if transformation == "uppercase":
            data[target_field] = str(value).upper()
        elif transformation == "lowercase":
            data[target_field] = str(value).lower()
        elif transformation == "trim":
            data[target_field] = str(value).strip()
        elif transformation == "remove_spaces":
            data[target_field] = str(value).replace(" ", "")
        
        return data
    
    def _apply_notation_logic(
        self,
        data: Dict[str, Any],
        rule: Dict
    ) -> Dict[str, Any]:
        """Apply notation-specific logic"""
        # Check if notation exists
        notation_exists = data.get("Notation_Exists", "").lower() == "yes"
        
        if notation_exists:
            # Set refused_qty to null if notation exists
            data["Refused_Qty"] = "null"
            logger.debug("Notation exists: setting Refused_Qty to null")
        
        return data
    
    def _parse_and_format_date(self, date_str: str, desired_format: str) -> str:
        """Parse date string and format to desired format"""
        # Common date formats to try
        formats = [
            "%m/%d/%Y",
            "%m/%d/%y",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%d/%m/%y",
            "%Y/%m/%d",
            "%m-%d-%Y",
            "%m-%d-%y",
            "%B %d, %Y",
            "%b %d, %Y",
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime(desired_format)
            except ValueError:
                continue
        
        # If no format matches, return original
        return date_str
    
    def apply_field_mapping(
        self,
        extracted_data: Dict[str, Any],
        template: Dict
    ) -> Dict[str, Any]:
        """Map extracted fields to target schema using template field_mapping"""
        field_mapping = template.get("field_mapping", {})
        
        if not field_mapping:
            return extracted_data
        
        mapped_data = {}
        
        for source_field, mapping_config in field_mapping.items():
            if source_field in extracted_data:
                target_field = mapping_config.get("target_field", source_field)
                data_type = mapping_config.get("type", "string")
                
                value = extracted_data[source_field]
                
                # Convert to target type
                try:
                    if data_type == "integer":
                        value = int(float(value)) if value not in [None, "null", ""] else 0
                    elif data_type == "float":
                        value = float(value) if value not in [None, "null", ""] else 0.0
                    elif data_type == "boolean":
                        value = str(value).lower() in ["yes", "true", "1"]
                    elif data_type == "string":
                        value = str(value) if value not in [None, "null"] else ""
                except Exception as e:
                    logger.warning(f"Error converting {source_field} to {data_type}: {e}")
                
                mapped_data[target_field] = value
        
        return mapped_data