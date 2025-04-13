import os
import logging
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from db_connector import DBConnector
from transpiler import MXQLTranspiler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MXQL")

class MXQL:
    """
    Main MXQL class that provides a high-level interface to the MXQL system
    """
    
    def __init__(self, db_connection_string: str, model_storage_path: Optional[str] = None):
        """
        Initialize MXQL
        
        Args:
            db_connection_string: SQLAlchemy connection string for the database
            model_storage_path: Path to store trained models
        """
        self.db_connector = DBConnector(db_connection_string)
        
        if model_storage_path is None:
            model_storage_path = os.path.join(os.getcwd(), "mxql_models")
        
        self.model_storage_path = model_storage_path
        os.makedirs(self.model_storage_path, exist_ok=True)
        
        self.transpiler = MXQLTranspiler(self.db_connector, self.model_storage_path)
    
    def execute(self, mxql_statement: str) -> Any:
        """
        Execute an MXQL statement
        
        Args:
            mxql_statement: MXQL statement as string
            
        Returns:
            Result of execution
        """
        return self.transpiler.execute(mxql_statement)
    
    def execute_file(self, file_path: str) -> List[Any]:
        """
        Execute MXQL statements from a file
        
        Args:
            file_path: Path to file containing MXQL statements
            
        Returns:
            List of execution results
        """
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Split the content by semicolons and filter empty statements
        statements = [stmt.strip() for stmt in content.split(';') if stmt.strip()]
        
        results = []
        for stmt in statements:
            if stmt:  # Skip empty statements
                results.append(self.execute(stmt + ';'))
        
        return results
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models
        
        Returns:
            List of model metadata
        """
        models = []
        
        if not os.path.exists(self.model_storage_path):
            return models
        
        for model_name in os.listdir(self.model_storage_path):
            model_dir = os.path.join(self.model_storage_path, model_name)
            if os.path.isdir(model_dir):
                metadata_path = os.path.join(model_dir, 'metadata.json')
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    models.append(metadata)
        
        return models
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model metadata or None if model doesn't exist
        """
        model_dir = os.path.join(self.model_storage_path, model_name)
        metadata_path = os.path.join(model_dir, 'metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if model was deleted, False otherwise
        """
        import shutil
        
        model_dir = os.path.join(self.model_storage_path, model_name)
        
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
            logger.info(f"Deleted model: {model_name}")
            return True
        else:
            logger.warning(f"Model not found: {model_name}")
            return False
    
    def explain(self, mxql_statement: str) -> str:
        """
        Explain what an MXQL statement will do without executing it
        
        Args:
            mxql_statement: MXQL statement as string
            
        Returns:
            Explanation of the statement
        """
        return self.transpiler.explain(mxql_statement)
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the database schema
        
        Returns:
            Dictionary containing database schema information
        """
        return self.db_connector.get_schema()
    
    def export_model(self, model_name: str, export_path: str) -> bool:
        """
        Export a model to a specified location
        
        Args:
            model_name: Name of the model to export
            export_path: Path to export the model to
            
        Returns:
            True if export was successful, False otherwise
        """
        import shutil
        
        model_dir = os.path.join(self.model_storage_path, model_name)
        
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            os.makedirs(export_path, exist_ok=True)
            export_file = os.path.join(export_path, f"{model_name}.mxql")
            
            # Create a zip file of the model directory
            shutil.make_archive(
                os.path.join(export_path, model_name),
                'zip',
                model_dir
            )
            
            logger.info(f"Exported model to: {export_file}.zip")
            return True
        else:
            logger.warning(f"Model not found: {model_name}")
            return False
    
    def import_model(self, import_path: str) -> bool:
        """
        Import a model from a specified location
        
        Args:
            import_path: Path to the model export file
            
        Returns:
            True if import was successful, False otherwise
        """
        import shutil
        import zipfile
        
        if not os.path.exists(import_path):
            logger.error(f"Import file not found: {import_path}")
            return False
        
        try:
            with zipfile.ZipFile(import_path, 'r') as zip_ref:
                # Extract the model name from the first directory in the zip
                model_name = zip_ref.namelist()[0].split('/')[0]
                model_dir = os.path.join(self.model_storage_path, model_name)
                
                # Create model directory and extract
                os.makedirs(model_dir, exist_ok=True)
                zip_ref.extractall(self.model_storage_path)
                
            logger.info(f"Imported model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to import model: {str(e)}")
            return False

def main():
    """
    Main entry point for the MXQL command-line interface
    """
    parser = argparse.ArgumentParser(description='MXQL - Machine Learning Query Language')
    parser.add_argument('--db', required=True, help='Database connection string')
    parser.add_argument('--model-path', help='Path to store ML models')
    parser.add_argument('--file', help='Execute MXQL statements from a file')
    parser.add_argument('--query', help='Execute a single MXQL query')
    parser.add_argument('--explain', action='store_true', help='Explain the query without executing it')
    parser.add_argument('--list-models', action='store_true', help='List all available models')
    parser.add_argument('--export-model', help='Export a model to the specified path')
    parser.add_argument('--import-model', help='Import a model from the specified path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    mxql = MXQL(args.db, args.model_path)
    
    if args.list_models:
        models = mxql.list_models()
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"  - {model['name']}: {model['description']}")
            print(f"    Type: {model['type']}")
            print(f"    Created: {model['created_at']}")
            print()
    
    elif args.export_model:
        if not args.file:
            print("Error: --file argument is required for export")
            return
        success = mxql.export_model(args.export_model, args.file)
        if success:
            print(f"Successfully exported model {args.export_model} to {args.file}")
        else:
            print(f"Failed to export model {args.export_model}")
    
    elif args.import_model:
        success = mxql.import_model(args.import_model)
        if success:
            print(f"Successfully imported model from {args.import_model}")
        else:
            print(f"Failed to import model from {args.import_model}")
    
    elif args.file:
        try:
            results = mxql.execute_file(args.file)
            for i, result in enumerate(results):
                print(f"Result {i+1}:")
                print(result)
                print()
        except Exception as e:
            print(f"Error executing file: {str(e)}")
    
    elif args.query:
        try:
            if args.explain:
                explanation = mxql.explain(args.query)
                print("Query explanation:")
                print(explanation)
            else:
                result = mxql.execute(args.query)
                print("Query result:")
                print(result)
        except Exception as e:
            print(f"Error executing query: {str(e)}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
