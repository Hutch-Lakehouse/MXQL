"""
Utility functions and shared code for MXQL modules
"""
import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def convert_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert query result to a pandas DataFrame
    
    Args:
        result: Query result that might need conversion
        
    Returns:
        pandas DataFrame
    
    Raises:
        ValueError: If the result cannot be converted to a DataFrame
    """
    if isinstance(result, pd.DataFrame):
        return result
    
    # If result is a list of dictionaries, convert to DataFrame
    if isinstance(result, list) and all(isinstance(item, dict) for item in result):
        return pd.DataFrame(result)
    
    raise ValueError("Query result cannot be converted to DataFrame")

def parse_query_string(query_string: str):
    """
    Parse a query string into an AST
    
    This function is used to avoid circular imports between parser and transformer.
    
    Args:
        query_string: The MXQL query string
        
    Returns:
        Parsed abstract syntax tree
    """
    from .mxql_parser import MXQLParser
    parser = MXQLParser()
    return parser.parse(query_string)
