#!/usr/bin/env python3
"""
Transformer module for standardizing and enriching parsed data.
This module handles data transformation operations.
"""
from typing import Dict, List, Any, Union, Optional
import re
import json
from datetime import datetime


class Transformer:
    """Transformer class for standardizing and enriching parsed data"""
    
    def __init__(self, config=None):
        """Initialize the transformer with optional configuration"""
        self.config = config or {}
        
    def transform(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform parsed data into standardized format
        
        Args:
            parsed_data: The parsed data structure from the Parser
            
        Returns:
            A dictionary with transformed and enriched data
        """
        transformed = {
            'metadata': self._extract_metadata(parsed_data),
            'content': self._process_content(parsed_data),
            'relationships': self._identify_relationships(parsed_data),
            'schema': self._infer_schema(parsed_data),
            'errors': self._validate_data(parsed_data)
        }
        
        return transformed
    
    def _extract_metadata(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and standardize metadata from parsed data"""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'sections_count': len(parsed_data.get('sections', [])),
            'lists_count': len(parsed_data.get('lists', [])),
            'code_blocks_count': len(parsed_data.get('code_blocks', [])),
            'tables_count': len(parsed_data.get('tables', []))
        }
        
        # Extract title if available from first section
        sections = parsed_data.get('sections', [])
        if sections and 'title' in sections[0]:
            metadata['title'] = sections[0]['title']
        
        # Add any explicit metadata from key-values
        key_values = parsed_data.get('key_values', {})
        metadata_keys = ['author', 'date', 'version', 'category', 'tags']
        for key in metadata_keys:
            if key in key_values:
                metadata[key] = key_values[key]
                
        # Process tags if they exist
        if 'tags' in metadata and isinstance(metadata['tags'], str):
            metadata['tags'] = [tag.strip() for tag in metadata['tags'].split(',')]
            
        return metadata
    
    def _process_content(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and standardize content from parsed data"""
        content = {
            'main_text': self._extract_main_text(parsed_data),
            'sections': self._normalize_sections(parsed_data.get('sections', [])),
            'structured_data': self._structure_data(parsed_data)
        }
        
        return content
    
    def _extract_main_text(self, parsed_data: Dict[str, Any]) -> str:
        """Extract main text content without markup"""
        sections = parsed_data.get('sections', [])
        if not sections:
            return ""
            
        main_text = []
        for section in sections:
            main_text.append(section.get('title', ''))
            main_text.append(section.get('content', ''))
            
        # Remove code blocks, tables, and other markup
        text = ' '.join(main_text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\|.*?\|', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _normalize_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize section structure"""
        normalized = []
        
        for section in sections:
            normalized_section = {
                'title': section.get('title', '').strip(),
                'level': section.get('level', 1),
                'content': section.get('content', '').strip(),
                'word_count': len(section.get('content', '').split()),
                'slug': self._create_slug(section.get('title', ''))
            }
            normalized.append(normalized_section)
            
        return normalized
    
    def _create_slug(self, text: str) -> str:
        """Create a URL-friendly slug from text"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s-]', '', text)
        text = re.sub(r'\s+', '-', text)
        return text.strip('-')
    
    def _structure_data(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parsed data into more structured formats"""
        structured = {}
        
        # Process tables into more accessible formats
        tables = parsed_data.get('tables', [])
        if tables:
            structured['tables'] = {}
            for i, table in enumerate(tables):
                headers = table.get('headers', [])
                rows = table.get('rows', [])
                
                table_name = f"table_{i+1}"
                structured['tables'][table_name] = {
                    'headers': headers,
                    'data': rows,
                    'row_count': len(rows)
                }
        
        # Process code blocks with language detection
        code_blocks = parsed_data.get('code_blocks', [])
        if code_blocks:
            structured['code'] = {}
            for i, block in enumerate(code_blocks):
                language = block.get('language', 'text')
                code = block.get('code', '')
                
                block_name = f"code_block_{i+1}"
                structured['code'][block_name] = {
                    'language': language,
                    'content': code,
                    'line_count': code.count('\n') + 1
                }
        
        return structured
    
    def _identify_relationships(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify relationships between different elements"""
        relationships = {
            'section_hierarchy': self._build_section_hierarchy(parsed_data.get('sections', [])),
            'references': self._extract_references(parsed_data)
        }
        
        return relationships
    
    def _build_section_hierarchy(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a hierarchical structure of sections based on their levels"""
        hierarchy = {}
        current_path = []
        
        for section in sections:
            level = section.get('level', 1)
            title = section.get('title', '')
            
            # Adjust the current path based on the section level
            if level <= len(current_path):
                current_path = current_path[:level-1]
            
            current_path.append(title)
            
            # Build the nested dictionary structure
            current = hierarchy
            for i, path_item in enumerate(current_path):
                if i == len(current_path) - 1:
                    current[path_item] = {'content': section.get('content', '')}
                else:
                    if path_item not in current:
                        current[path_item] = {}
                    current = current[path_item]
        
        return hierarchy
    
    def _extract_references(self, parsed_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract references and links from the content"""
        references = {
            'urls': [],
            'cross_references': []
        }
        
        # Extract URLs
        all_text = self._extract_main_text(parsed_data)
        url_pattern = re.compile(r'https?://\S+')
        references['urls'] = url_pattern.findall(all_text)
        
        # Extract potential cross-references (like "[Section X]")
        cross_ref_pattern = re.compile(r'\[(.*?)\]')
        references['cross_references'] = cross_ref_pattern.findall(all_text)
        
        return references
    
    def _infer_schema(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Infer schema from structured data elements"""
        schema = {}
        
        # Infer schema from tables
        tables = parsed_data.get('tables', [])
        if tables:
            schema['tables'] = {}
            for i, table in enumerate(tables):
                headers = table.get('headers', [])
                rows = table.get('rows', [])
                
                table_name = f"table_{i+1}"
                field_types = {}
                
                # Analyze first few rows to infer types
                for header in headers:
                    field_types[header] = self._infer_field_type(rows, header)
                
                schema['tables'][table_name] = field_types
        
        # Infer schema from key-values
        key_values = parsed_data.get('key_values', {})
        if key_values:
            schema['properties'] = {}
            for key, value in key_values.items():
                schema['properties'][key] = self._infer_value_type(value)
        
        return schema
    
    def _infer_field_type(self, rows: List[Dict[str, str]], field: str) -> str:
        """Infer the data type of a field based on its values"""
        if not rows:
            return 'string'
            
        types = []
        for row in rows[:min(5, len(rows))]:  # Check first 5 rows or fewer
            if field in row:
                types.append(self._infer_value_type(row[field]))
        
        if not types:
            return 'string'
            
        # Return most common type
        if all(t == 'integer' for t in types):
            return 'integer'
        if all(t in ('integer', 'float') for t in types):
            return 'float'
        if all(t == 'boolean' for t in types):
            return 'boolean'
        if all(t == 'date' for t in types):
            return 'date'
            
        return 'string'  # Default type
    
    def _infer_value_type(self, value: str) -> str:
        """Infer the data type of a single value"""
        if not value or not isinstance(value, str):
            return 'string'
            
        # Check for boolean
        if value.lower() in ('true', 'false', 'yes', 'no'):
            return 'boolean'
            
        # Check for integer
        try:
            int(value)
            return 'integer'
        except ValueError:
            pass
            
        # Check for float
        try:
            float(value)
            return 'float'
        except ValueError:
            pass
            
        # Check for date
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}\.\d{2}\.\d{4}'  # DD.MM.YYYY
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, value):
                return 'date'
                
        return 'string'
    
    def _validate_data(self, parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate parsed data and return any errors or warnings"""
        errors = []
        
        # Check for missing sections
        if not parsed_data.get('sections'):
            errors.append({
                'type': 'warning',
                'message': 'No sections found in the document',
                'element': 'sections'
            })
        
        # Validate tables structure
        tables = parsed_data.get('tables', [])
        for i, table in enumerate(tables):
            headers = table.get('headers', [])
            rows = table.get('rows', [])
            
            for j, row in enumerate(rows):
                for header in headers:
                    if header not in row:
                        errors.append({
                            'type': 'error',
                            'message': f'Missing data for header "{header}" in table {i+1}, row {j+1}',
                            'element': f'tables.{i}.rows.{j}.{header}'
                        })
                        
        return errors


if __name__ == "__main__":
    # Simple test
    from parser import Parser
    
    sample_text = """# Sample Document
    
    Author: John Doe
    Date: 2023-04-15
    
    ## Introduction
    
    This is an introduction.
    
    ## Data
    
    | Name | Age | Email |
    |------|-----|-------|
    | John | 30  | john@example.com |
    | Jane | 25  | jane@example.com |
    
    ## Code Example
    
    ```python
    def hello():
        print("Hello, world!")
    ```
    """
    
    parser = Parser()
    parsed_data = parser.parse(sample_text)
    
    transformer = Transformer()
    transformed_data = transformer.transform(parsed_data)
    
    print(json.dumps(transformed_data, indent=2))
