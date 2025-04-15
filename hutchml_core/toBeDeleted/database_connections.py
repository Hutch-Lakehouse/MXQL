#!/usr/bin/env python3
"""
Database connection module for storing and retrieving parsed data.
This module handles database operations.
"""
import sqlite3
import json
from typing import Dict, List, Any, Union, Optional
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection handler for parsed data"""
    
    def __init__(self, db_path="parsed_data.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Establish connection to the database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
            self._initialize_tables()
            return True
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            return False
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        # Documents table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Sections table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            title TEXT,
            content TEXT,
            level INTEGER,
            position INTEGER,
            FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
        )
        ''')
        
        # Tables table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            table_name TEXT,
            headers TEXT,
            data TEXT,
            FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
        )
        ''')
        
        # Code blocks table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS code_blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            language TEXT,
            code TEXT,
            position INTEGER,
            FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
        )
        ''')
        
        # Key-value pairs table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS key_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            key TEXT,
            value TEXT,
            FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
        )
        ''')
        
        self.conn.commit()
        logger.info("Database tables initialized")
    
    def store_document(self, parsed_data: Dict[str, Any], transformed_data: Dict[str, Any] = None) -> int:
        """
        Store parsed document and its components in the database
        
        Args:
            parsed_data: The parsed data structure
            transformed_data: Optional transformed data
            
        Returns:
            The ID of the inserted document
        """
        if not self.conn:
            self.connect()
        
        try:
            # Get metadata
            metadata = {}
            if transformed_data and 'metadata' in transformed_data:
                metadata = transformed_data['metadata']
            
            # Get title
            title = metadata.get('title', '')
            if not title and 'sections' in parsed_data and parsed_data['sections']:
                title = parsed_data['sections'][0].get('title', '')
            
            # Get content
            content = ""
            if transformed_data and 'content' in transformed_data:
                content = transformed_data['content'].get('main_text', '')
            
            # Insert document
            self.cursor.execute('''
            INSERT INTO documents (title, content, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                title,
                content,
                json.dumps(metadata),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            document_id = self.cursor.lastrowid
            
            # Store sections
            self._store_sections(document_id, parsed_data.get('sections', []))
            
            # Store tables
            self._store_tables(document_id, parsed_data.get('tables', []))
            
            # Store code blocks
            self._store_code_blocks(document_id, parsed_data.get('code_blocks', []))
            
            # Store key-value pairs
            self._store_key_values(document_id, parsed_data.get('key_values', {}))
            
            self.conn.commit()
            logger.info(f"Document stored with ID: {document_id}")
            
            return document_id
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Error storing document: {e}")
            return -1
    
    def _store_sections(self, document_id: int, sections: List[Dict[str, Any]]):
        """Store document sections"""
        for position, section in enumerate(sections):
            self.cursor.execute('''
            INSERT INTO sections (document_id, title, content, level, position)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                document_id,
                section.get('title', ''),
                section.get('content', ''),
                section.get('level', 1),
                position
            ))
    
    def _store_tables(self, document_id: int, tables: List[Dict[str, Any]]):
        """Store document tables"""
        for i, table in enumerate(tables):
            table_name = f"table_{i+1}"
            headers = table.get('headers', [])
            rows = table.get('rows', [])
            
            self.cursor.execute('''
            INSERT INTO data_tables (document_id, table_name, headers, data)
            VALUES (?, ?, ?, ?)
            ''', (
                document_id,
                table_name,
                json.dumps(headers),
                json.dumps(rows)
            ))
    
    def _store_code_blocks(self, document_id: int, code_blocks: List[Dict[str, Any]]):
        """Store document code blocks"""
        for position, block in enumerate(code_blocks):
            self.cursor.execute('''
            INSERT INTO code_blocks (document_id, language, code, position)
            VALUES (?, ?, ?, ?)
            ''', (
                document_id,
                block.get('language', 'text'),
                block.get('code', ''),
                position
            ))
    
    def _store_key_values(self, document_id: int, key_values: Dict[str, str]):
        """Store document key-value pairs"""
        for key, value in key_values.items():
            self.cursor.execute('''
            INSERT INTO key_values (document_id, key, value)
            VALUES (?, ?, ?)
            ''', (
                document_id,
                key,
                value
            ))
    
    def get_document(self, document_id: int) -> Dict[str, Any]:
        """
        Retrieve a document and all its components by ID
        
        Args:
            document_id: The ID of the document to retrieve
            
        Returns:
            A dictionary containing the document and its components
        """
        if not self.conn:
            self.connect()
        
        try:
            # Get document
            self.cursor.execute('SELECT * FROM documents WHERE id = ?', (document_id,))
            document = dict(self.cursor.fetchone())
            
            # Parse metadata JSON
            document['metadata'] = json.loads(document['metadata'])
            
            # Get sections
            self.cursor.execute('''
            SELECT id, title, content, level, position
            FROM sections
            WHERE document_id = ?
            ORDER BY position
            ''', (document_id,))
            document['sections'] = [dict(row) for row in self.cursor.fetchall()]
            
            # Get tables
            self.cursor.execute('SELECT * FROM data_tables WHERE document_id = ?', (document_id,))
            tables = []
            for row in self.cursor.fetchall():
                table_data = dict(row)
                table_data['headers'] = json.loads(table_data['headers'])
                table_data['data'] = json.loads(table_data['data'])
                tables.append(table_data)
            document['tables'] = tables
            
            # Get code blocks
            self.cursor.execute('''
            SELECT id, language, code, position
            FROM code_blocks
            WHERE document_id = ?
            ORDER BY position
            ''', (document_id,))
            document['code_blocks'] = [dict(row) for row in self.cursor.fetchall()]
            
            # Get key-value pairs
            self.cursor.execute('SELECT key, value FROM key_values WHERE document_id = ?', (document_id,))
            key_values = {}
            for row in self.cursor.fetchall():
                key_values[row['key']] = row['value']
            document['key_values'] = key_values
            
            return document
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving document: {e}")
            return {}
    
    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query
        
        Args:
            query: The search query
            
        Returns:
            A list of matching documents
        """
        if not self.conn:
            self.connect()
        
        try:
            # Search in title, content, and metadata
            self.cursor.execute('''
            SELECT id, title, substr(content, 1, 200) as content_preview, created_at
            FROM documents
            WHERE title LIKE ? OR content LIKE ? OR metadata LIKE ?
            ORDER BY created_at DESC
            ''', (f'%{query}%', f'%{query}%', f'%{query}%'))
            
            results = [dict(row) for row in self.cursor.fetchall()]
            
            # Add match count
            for result in results:
                title_matches = result['title'].lower().count(query.lower())
                content_matches = result.get('content_preview', '').lower().count(query.lower())
                result['relevance'] = title_matches * 2 + content_matches
            
            # Sort by relevance
            results.sort(key=lambda x: x['relevance'], reverse=True)
            
            return results
            
        except sqlite3.Error as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def update_document(self, document_id: int, parsed_data: Dict[str, Any], transformed_data: Dict[str, Any] = None) -> bool:
        """
        Update an existing document
        
        Args:
            document_id: The ID of the document to update
            parsed_data: The new parsed data
            transformed_data: Optional new transformed data
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self.conn:
            self.connect()
        
        try:
            # Begin transaction
            self.conn.execute('BEGIN TRANSACTION')
            
            # Delete existing document components
            self.cursor.execute('DELETE FROM sections WHERE document_id = ?', (document_id,))
            self.cursor.execute('DELETE FROM data_tables WHERE document_id = ?', (document_id,))
            self.cursor.execute('DELETE FROM code_blocks WHERE document_id = ?', (document_id,))
            self.cursor.execute('DELETE FROM key_values WHERE document_id = ?', (document_id,))
            
            # Get metadata
            metadata = {}
            if transformed_data and 'metadata' in transformed_data:
                metadata = transformed_data['metadata']
            
            # Get title
            title = metadata.get('title', '')
            if not title and 'sections' in parsed_data and parsed_data['sections']:
                title = parsed_data['sections'][0].get('title', '')
            
            # Get content
            content = ""
            if transformed_data and 'content' in transformed_data:
                content = transformed_data['content'].get('main_text', '')
            
            # Update document
            self.cursor.execute('''
            UPDATE documents
            SET title = ?, content = ?, metadata = ?, updated_at = ?
            WHERE id = ?
            ''', (
                title,
                content,
                json.dumps(metadata),
                datetime.now().isoformat(),
                document_id
            ))
            
            # Store new components
            self._store_sections(document_id, parsed_data.get('sections', []))
            self._store_tables(document_id, parsed_data.get('tables', []))
            self._store_code_blocks(document_id, parsed_data.get('code_blocks', []))
            self._store_key_values(document_id, parsed_data.get('key_values', {}))
            
            self.conn.commit()
            logger.info(f"Document updated with ID: {document_id}")
            
            return True
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Error updating document: {e}")
            return False
    
    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document and all its components
        
        Args:
            document_id: The ID of the document to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self.conn:
            self.connect()
        
        try:
            # Delete document (cascades to all related components)
            self.cursor.execute('DELETE FROM documents WHERE id = ?', (document_id,))
            self.conn.commit()
            
            if self.cursor.rowcount > 0:
                logger.info(f"Document deleted with ID: {document_id}")
                return True
            else:
                logger.warning(f"No document found with ID: {document_id}")
                return False
                
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Error deleting document: {e}")
            return False
    
    def export_document(self, document_id: int, format_type: str = "json") -> str:
        """
        Export a document in the specified format
        
        Args:
            document_id: The ID of the document to export
            format_type: The export format (json, markdown, html)
            
        Returns:
            The exported document as a string
        """
        document = self.get_document(document_id)
        
        if not document:
            return ""
        
        if format_type == "json":
            return json.dumps(document, indent=2)
        
        # For other formats, we'll use the transpiler later
        return json.dumps(document, indent=2)  # Default to JSON


if __name__ == "__main__":
    # Simple test
    db = DatabaseConnection("test.db")
    db.connect()
    
    # Sample data
    sample_data = {
        "sections": [
            {
                "title": "Sample Document",
                "content": "This is a sample document.",
                "level": 1
            },
            {
                "title": "Introduction",
                "content": "This is the introduction.",
                "level": 2
            }
        ],
        "key_values": {
            "author": "John Doe",
            "date": "2023-04-15"
        },
        "code_blocks": [
            {
                "language": "python",
                "code": "print('Hello, world!')"
            }
        ],
        "tables": [
            {
                "headers": ["Name", "Age"],
                "rows": [
                    {"Name": "John", "Age": "30"},
                    {"Name": "Jane", "Age": "25"}
                ]
            }
        ]
    }
    
    # Store document
    doc_id = db.store_document(sample_data)
    
    # Retrieve document
    retrieved = db.get_document(doc_id)
    print(json.dumps(retrieved, indent=2))
    
    # Close connection
    db.close()
