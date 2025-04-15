#!/usr/bin/env python3
"""
Parser module for extracting structured data from input text.
This module focuses solely on parsing logic.
"""
import re
from typing import Dict, List, Any, Union, Optional

class Parser:
    """Parser class for converting input text to structured data"""
    
    def __init__(self, config=None):
        """Initialize the parser with optional configuration"""
        self.config = config or {}
        self.patterns = {
            'section': re.compile(r'^#+\s+(.+)$', re.MULTILINE),
            'list_item': re.compile(r'^\s*[-*]\s+(.+)$', re.MULTILINE),
            'key_value': re.compile(r'^([^:]+):\s*(.+)$', re.MULTILINE),
            'code_block': re.compile(r'```([a-z]*)\n(.*?)\n```', re.DOTALL),
            'table': re.compile(r'\|(.+)\|\n\|[-|]+\|\n((?:\|.+\|\n)+)', re.MULTILINE)
        }
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse input text into a structured data dictionary
        
        Args:
            text: The input text to parse
            
        Returns:
            A dictionary containing the parsed data structure
        """
        result = {
            'sections': self._parse_sections(text),
            'key_values': self._parse_key_values(text),
            'lists': self._parse_lists(text),
            'code_blocks': self._parse_code_blocks(text),
            'tables': self._parse_tables(text)
        }
        
        return result
    
    def _parse_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract sections and their content"""
        sections = []
        matches = self.patterns['section'].finditer(text)
        
        last_pos = 0
        for match in matches:
            section_title = match.group(1).strip()
            start_pos = match.end()
            
            # Find the next section or end of text
            next_match = self.patterns['section'].search(text, start_pos)
            end_pos = next_match.start() if next_match else len(text)
            
            section_content = text[start_pos:end_pos].strip()
            sections.append({
                'title': section_title,
                'content': section_content,
                'level': len(match.group(0)) - len(match.group(0).lstrip('#'))
            })
            
            last_pos = end_pos
        
        return sections
    
    def _parse_lists(self, text: str) -> List[str]:
        """Extract list items"""
        list_items = []
        matches = self.patterns['list_item'].finditer(text)
        
        for match in matches:
            list_items.append(match.group(1).strip())
        
        return list_items
    
    def _parse_key_values(self, text: str) -> Dict[str, str]:
        """Extract key-value pairs"""
        key_values = {}
        matches = self.patterns['key_value'].finditer(text)
        
        for match in matches:
            key = match.group(1).strip()
            value = match.group(2).strip()
            key_values[key] = value
        
        return key_values
    
    def _parse_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks with language info"""
        code_blocks = []
        matches = self.patterns['code_block'].finditer(text)
        
        for match in matches:
            language = match.group(1).strip() or 'text'
            code = match.group(2).strip()
            code_blocks.append({
                'language': language,
                'code': code
            })
        
        return code_blocks
    
    def _parse_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract tables with headers and rows"""
        tables = []
        matches = self.patterns['table'].finditer(text)
        
        for match in matches:
            header_row = match.group(1).strip()
            headers = [cell.strip() for cell in header_row.split('|') if cell.strip()]
            
            data_rows = match.group(2).strip().split('\n')
            rows = []
            
            for row in data_rows:
                cells = [cell.strip() for cell in row.split('|') if cell.strip()]
                row_data = {}
                
                for i, header in enumerate(headers):
                    if i < len(cells):
                        row_data[header] = cells[i]
                
                rows.append(row_data)
            
            tables.append({
                'headers': headers,
                'rows': rows
            })
        
        return tables


if __name__ == "__main__":
    # Simple test
    sample_text = """# Sample Section
    
    This is some content.
    
    - List item 1
    - List item 2
    
    key1: value1
    key2: value2
    
    ```python
    def hello():
        print("Hello, world!")
    ```
    
    | Name | Age |
    |------|-----|
    | John | 30  |
    | Jane | 25  |
    """
    
    parser = Parser()
    result = parser.parse(sample_text)
    print(result)
