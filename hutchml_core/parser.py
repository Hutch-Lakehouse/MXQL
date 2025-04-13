import os
import logging
from typing import Optional
from lark import Lark
from lark.exceptions import UnexpectedToken, UnexpectedCharacters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MXQL.Parser")

class MXQLParser:
    """
    Parser for MXQL language using Lark grammar
    """
    
    def __init__(self, grammar_path: Optional[str] = None):
        """
        Initialize the MXQL parser
        
        Args:
            grammar_path: Path to the MXQL grammar file (.lark)
        """
        if grammar_path is None:
            # Use default grammar file path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            grammar_path = os.path.join(current_dir, "mxql.lark")
        
        if not os.path.exists(grammar_path):
            raise FileNotFoundError(f"Grammar file not found: {grammar_path}")
        
        with open(grammar_path, 'r') as f:
            grammar = f.read()
        
        # Initialize the Lark parser with the MXQL grammar
        self.parser = Lark(grammar, start='statement', parser='lalr')
        logger.info(f"Initialized MXQL parser with grammar from {grammar_path}")
    
    def parse(self, mxql_statement: str):
        """
        Parse an MXQL statement
        
        Args:
            mxql_statement: MXQL statement as string
            
        Returns:
            Parsed abstract syntax tree
        """
        try:
            return self.parser.parse(mxql_statement)
        except (UnexpectedToken, UnexpectedCharacters) as e:
            error_message = f"Syntax error in MXQL statement: {str(e)}"
            logger.error(error_message)
            raise SyntaxError(error_message)
        except Exception as e:
            error_message = f"Error parsing MXQL statement: {str(e)}"
            logger.error(error_message)
            raise ValueError(error_message)
