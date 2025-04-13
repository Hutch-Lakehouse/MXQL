
import os
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime

from lark import Transformer
from db_connector import DBConnector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MXQL.Transformer")

class MXQLTransformer(Transformer):
    """
    Transformer for MXQL AST to executable operations
    """
    
    def __init__(self, db_connector: DBConnector, model_storage_path: str):
        """
        Initialize the MXQL transformer
        
        Args:
            db_connector: Database connector instance
            model_storage_path: Path to store ML models
        """
        super().__init__()
        self.db_connector = db_connector
        self.model_storage_path = model_storage_path
        
        # Import ML libraries lazily to avoid unnecessary dependencies
        self._ml_modules = {}
    
    def _get_ml_module(self, module_type: str):
        """
        Get ML module lazily
        
        Args:
            module_type: Type of ML module ('sklearn', 'pytorch', etc.)
            
        Returns:
            ML module
        """
        if module_type not in self._ml_modules:
            if module_type == 'sklearn':
                try:
                    import sklearn
                    self._ml_modules['sklearn'] = sklearn
                except ImportError:
                    raise ImportError("scikit-learn is required for ML operations but not installed")
            elif module_type == 'pytorch':
                try:
                    import torch
                    self._ml_modules['pytorch'] = torch
                except ImportError:
                    raise ImportError("PyTorch is required for ML operations but not installed")
            else:
                raise ValueError(f"Unsupported ML module type: {module_type}")
        
        return self._ml_modules[module_type]
    
    # Statement handlers
    
    def statement(self, items):
        # The main statement handler that routes to specific statement types
        if len(items) == 1:
            return items[0]
        else:
            logger.warning(f"Multiple items in statement: {items}")
            return items
    
    def create_model_stmt(self, items):
        model_name = items[0].value
        model_type = items[1].value
        options = {}
        
        # Parse options if provided
        if len(items) > 2:
            options = items[2]
        
        # Handle different model types
        if model_type.lower() == 'classifier':
            return self._create_classifier(model_name, options)
        elif model_type.lower() == 'regressor':
            return self._create_regressor(model_name, options)
        elif model_type.lower() == 'cluster':
            return self._create_cluster(model_name, options)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train_model_stmt(self, items):
        model_name = items[0].value
        query = items[1] if len(items) > 1 else None
        options = {}
        
        # Parse options if provided
        if len(items) > 2:
            options = items[2]
        
        return self._train_model(model_name, query, options)
    
    def predict_stmt(self, items):
        model_name = items[0].value
        query = items[1] if len(items) > 1 else None
        options = {}
        
        # Parse options if provided
        if len(items) > 2:
            options = items[2]
        
        return self._predict(model_name, query, options)
    
    def select_stmt(self, items):
        columns = items[0]
        table = items[1].value
        where_clause = items[2] if len(items) > 2 else None
        
        return self._execute_select(columns, table, where_clause)
    
    def insert_stmt(self, items):
        table = items[0].value
        columns = items[1]
        values = items[2]
        
        return self._execute_insert(table, columns, values)
    
    def update_stmt(self, items):
        table = items[0].value
        set_clause = items[1]
        where_clause = items[2] if len(items) > 2 else None
        
        return self._execute_update(table, set_clause, where_clause)
    
    def delete_stmt(self, items):
        table = items[0].value
        where_clause = items[1] if len(items) > 1 else None
        
        return self._execute_delete(table, where_clause)
    
    def options(self, items):
        result = {}
        for option in items:
            key, value = option
            result[key] = value
        return result
    
    def option(self, items):
        return (items[0].value, items[1])
    
    def option_value(self, items):
        # Handle different types of option values
        item = items[0]
        if hasattr(item, 'value'):
            # This is a token
            if item.type == 'NUMBER':
                return float(item.value)
            elif item.type == 'STRING':
                return item.value[1:-1]  # Remove quotes
            else:
                return item.value
        else:
            # This is a processed value
            return item
    
    def column_list(self, items):
        return [item.value if hasattr(item, 'value') else item for item in items]
    
    def where_clause(self, items):
        return items[0]
    
    def condition(self, items):
        if len(items) == 3:
            left, op, right = items
            return {
                'left': left.value if hasattr(left, 'value') else left,
                'operator': op.value if hasattr(op, 'value') else op,
                'right': right.value if hasattr(right, 'value') else right
            }
        elif len(items) == 1:
            # Simple condition or nested condition
            return items[0]
    
    def and_condition(self, items):
        return {
            'operator': 'AND',
            'conditions': items
        }
    
    def or_condition(self, items):
        return {
            'operator': 'OR',
            'conditions': items
        }
    
    # Implementation methods
    
    def _create_classifier(self, model_name: str, options: Dict[str, Any]):
        """Create a classifier model"""
        
        # Default to sklearn's Random Forest if not specified
        algorithm = options.get('algorithm', 'random_forest')
        
        # Create model directory
        model_dir = os.path.join(self.model_storage_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'name': model_name,
            'type': 'classifier',
            'algorithm': algorithm,
            'created_at': datetime.now().isoformat(),
            'trained': False,
            'options': options
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created classifier model '{model_name}' using {algorithm}")
        return {
            'status': 'success',
            'message': f"Created classifier model '{model_name}'",
            'model': metadata
        }
    
    def _create_regressor(self, model_name: str, options: Dict[str, Any]):
        """Create a regressor model"""
        
        # Default to sklearn's Linear Regression if not specified
        algorithm = options.get('algorithm', 'linear_regression')
        
        # Create model directory
        model_dir = os.path.join(self.model_storage_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'name': model_name,
            'type': 'regressor',
            'algorithm': algorithm,
            'created_at': datetime.now().isoformat(),
            'trained': False,
            'options': options
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created regressor model '{model_name}' using {algorithm}")
        return {
            'status': 'success',
            'message': f"Created regressor model '{model_name}'",
            'model': metadata
        }
    
    def _create_cluster(self, model_name: str, options: Dict[str, Any]):
        """Create a clustering model"""
        
        # Default to sklearn's KMeans if not specified
        algorithm = options.get('algorithm', 'kmeans')
        
        # Create model directory
        model_dir = os.path.join(self.model_storage_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'name': model_name,
            'type': 'cluster',
            'algorithm': algorithm,
            'created_at': datetime.now().isoformat(),
            'trained': False,
            'options': options
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created clustering model '{model_name}' using {algorithm}")
        return {
            'status': 'success',
            'message': f"Created clustering model '{model_name}'",
            'model': metadata
        }
    
    def _train_model(self, model_name: str, query, options: Dict[str, Any]):
        """Train a model using data from query"""
        
        # Get model metadata
        model_dir = os.path.join(self.model_storage_path, model_name)
        metadata_path = os.path.join(model_dir, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            raise ValueError(f"Model '{model_name}' does not exist")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Execute query to get training data
        if query:
            data = self._execute_query(query)
        else:
            raise ValueError("Training data query is required")
        
        # Extract features and target
        target_column = options.get('target', None)
        if not target_column and metadata['type'] in ['classifier', 'regressor']:
            raise ValueError("Target column must be specified for classifier/regressor")
        
        feature_columns = options.get('features', None)
        if not feature_columns:
            # Use all columns except target
            if target_column:
                feature_columns = [col for col in data.columns if col != target_column]
            else:
                feature_columns = list(data.columns)
        
        # Train the model
        model_type = metadata['type']
        algorithm = metadata['algorithm']
        
        try:
            if model_type == 'classifier':
                self._train_classifier(model_name, data, feature_columns, target_column, algorithm, options)
            elif model_type == 'regressor':
                self._train_regressor(model_name, data, feature_columns, target_column, algorithm, options)
            elif model_type == 'cluster':
                self._train_cluster(model_name, data, feature_columns, algorithm, options)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Update metadata
            metadata['trained'] = True
            metadata['trained_at'] = datetime.now().isoformat()
            metadata['feature_columns'] = feature_columns
            if target_column:
                metadata['target_column'] = target_column
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Trained {model_type} model '{model_name}'")
            return {
                'status': 'success',
                'message': f"Trained {model_type} model '{model_name}'",
                'model': metadata
            }
            
        except Exception as e:
            logger.error(f"Error training model '{model_name}': {str(e)}")
            raise
    
    def _train_classifier(self, model_name: str, data, feature_columns, target_column, algorithm, options):
        """Train a classifier model"""
        sklearn = self._get_ml_module('sklearn')
        
        X = data[feature_columns]
        y = data[target_column]
        
        # Create the classifier based on algorithm
        if algorithm == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=options.get('n_estimators', 100),
                max_depth=options.get('max_depth', None),
                random_state=options.get('random_state', 42)
            )
        elif algorithm == 'svm':
            from sklearn.svm import SVC
            model = SVC(
                C=options.get('C', 1.0),
                kernel=options.get('kernel', 'rbf'),
                random_state=options.get('random_state', 42)
            )
        elif algorithm == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(
                C=options.get('C', 1.0),
                max_iter=options.get('max_iter', 100),
                random_state=options.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unsupported classifier algorithm: {algorithm}")
        
        # Train the model
        model.fit(X, y)
        
        # Save the model
        import pickle
        model_path = os.path.join(self.model_storage_path, model_name, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model
    
    def _train_regressor(self, model_name: str, data, feature_columns, target_column, algorithm, options):
        """Train a regressor model"""
        sklearn = self._get_ml_module('sklearn')
        
        X = data[feature_columns]
        y = data[target_column]
        
        # Create the regressor based on algorithm
        if algorithm == 'linear_regression':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression(
                fit_intercept=options.get('fit_intercept', True),
                n_jobs=options.get('n_jobs', None)
            )
        elif algorithm == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=options.get('n_estimators', 100),
                max_depth=options.get('max_depth', None),
                random_state=options.get('random_state', 42)
            )
        elif algorithm == 'svr':
            from sklearn.svm import SVR
            model = SVR(
                C=options.get('C', 1.0),
                kernel=options.get('kernel', 'rbf')
            )
        else:
            raise ValueError(f"Unsupported regressor algorithm: {algorithm}")
        
        # Train the model
        model.fit(X, y)
        
        # Save the model
        import pickle
        model_path = os.path.join(self.model_storage_path, model_name, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model
    
    def _train_cluster(self, model_name: str, data, feature_columns, algorithm, options):
        """Train a clustering model"""
        sklearn = self._get_ml_module('sklearn')
        
        X = data[feature_columns]
        
        # Create the clustering model based on algorithm
        if algorithm == 'kmeans':
            from sklearn.cluster import KMeans
            model = KMeans(
                n_clusters=options.get('n_clusters', 8),
                random_state=options.get('random_state', 42),
                n_init=options.get('n_init', 10)
            )
        elif algorithm == 'dbscan':
            from sklearn.cluster import DBSCAN
            model = DBSCAN(
                eps=options.get('eps', 0.5),
                min_samples=options.get('min_samples', 5)
            )
        elif algorithm == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            model = AgglomerativeClustering(
                n_clusters=options.get('n_clusters', 8),
                linkage=options.get('linkage', 'ward')
            )
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
        
        # Train the model
        model.fit(X)
        
        # Save the model
        import pickle
        model_path = os.path.join(self.model_storage_path, model_name, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model
    
    def _predict(self, model_name: str, query, options: Dict[str, Any]):
        """Make predictions using a trained model"""
        
        # Get model metadata
        model_dir = os.path.join(self.model_storage_path, model_name)
        metadata_path = os.path.join(model_dir, 'metadata.json')
        model_path = os.path.join(model_dir, 'model.pkl')
        
        if not os.path.exists(metadata_path):
            raise ValueError(f"Model '{model_name}' does not exist")
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model '{model_name}' has not been trained yet")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load the model
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Execute query to get prediction data
        if query:
            data = self._execute_query(query)
        else:
            raise ValueError("Prediction data query is required")
        
        # Get feature columns from metadata
        feature_columns = metadata.get('feature_columns')
        if not feature_columns:
            raise ValueError("Feature columns not found in model metadata")
        
        # Make predictions
        model_type = metadata['type']
        
        try:
            X = data[feature_columns]
            
            if model_type == 'classifier':
                predictions = model.predict(X)
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)
                    # Create result DataFrame with predictions and probabilities
                    import pandas as pd
                    result = pd.DataFrame({
                        'prediction': predictions
                    })
                    class_columns = {f'probability_class_{i}': prob for i, prob in enumerate(probabilities.T)}
                    for col, values in class_columns.items():
                        result[col] = values
                else:
                    import pandas as pd
                    result = pd.DataFrame({
                        'prediction': predictions
                    })
            
            elif model_type == 'regressor':
                predictions = model.predict(X)
                import pandas as pd
                result = pd.DataFrame({
                    'prediction': predictions
                })
            
            elif model_type == 'cluster':
                clusters = model.fit_predict(X)
                import pandas as pd
                result = pd.DataFrame({
                    'cluster': clusters
                })
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Add original data if requested
            if options.get('include_original', False):
                result = pd.concat([data, result], axis=1)
            
            logger.info(f"Made predictions with {model_type} model '{model_name}'")
            
            # Convert result to dict for JSON serialization
            return result.to_dict(orient='records')
            
        except Exception as e:
            logger.error(f"Error making predictions with model '{model_name}': {str(e)}")
            raise
    
    def _execute_select(self, columns, table, where_clause):
        """Execute a SELECT query"""
        return self.db_connector.execute_select(columns, table, where_clause)
    
    def _execute_insert(self, table, columns, values):
        """Execute an INSERT query"""
        return self.db_connector.execute_insert(table, columns, values)
    
    def _execute_update(self, table, set_clause, where_clause):
        """Execute an UPDATE query"""
        return self.db_connector.execute_update(table, set_clause, where_clause)
    
    def _execute_delete(self, table, where_clause):
        """Execute a DELETE query"""
        return self.db_connector.execute_delete(table, where_clause)
    
    def _execute_query(self, query):
        """Execute a query and return the results as a pandas DataFrame"""
        # If the query is a string, parse it
        if isinstance(query, str):
            from mxql_parser import MXQLParser
            parser = MXQLParser()
            query = parser.parse(query)
        
        # Execute the query
        result = self.transform(query)
        
        # Convert result to DataFrame if it's not already
        import pandas as pd
        if not isinstance(result, pd.DataFrame):
            # If result is a list of dictionaries, convert to DataFrame
            if isinstance(result, list) and all(isinstance(item, dict) for item in result):
                result = pd.DataFrame(result)
            else:
                raise ValueError("Query result cannot be converted to DataFrame")
        
        return result
