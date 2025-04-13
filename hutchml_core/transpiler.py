from lark import Lark, Transformer, v_args
import json
import os
import tempfile
import uuid
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MXQLTranspiler")

class MXQLTranspiler:
    def __init__(self, db_connector, model_storage_path=None):
        """
        Initialize the MXQL Transpiler
        
        Args:
            db_connector: Database connector instance to interact with SQL database
            model_storage_path: Path to store trained models
        """
        self.db_connector = db_connector
        self.model_storage_path = model_storage_path or os.path.join(os.getcwd(), "mxql_models")
        
        # Ensure model storage directory exists
        os.makedirs(self.model_storage_path, exist_ok=True)
        
        # Load grammar from file
        grammar_path = Path(__file__).parent / "mxql.lark"
        with open(grammar_path, "r") as f:
            self.grammar = f.read()
        
        self.parser = Lark(self.grammar, parser='lalr', transformer=MXQLTransformer())
    
    def transpile(self, mxql_statement):
        """
        Parse and transpile an MXQL statement to Python code
        
        Args:
            mxql_statement: MXQL statement as string
            
        Returns:
            Generated Python code as string
        """
        try:
            tree = self.parser.parse(mxql_statement)
            return self._generate_code(tree)
        except Exception as e:
            logger.error(f"Error transpiling MXQL: {e}")
            raise
    
    def execute(self, mxql_statement):
        """
        Execute an MXQL statement
        
        Args:
            mxql_statement: MXQL statement as string
            
        Returns:
            Result of execution
        """
        python_code = self.transpile(mxql_statement)
        
        # Create temporary file for the generated code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            f.write(python_code)
            temp_file_path = f.name
        
        logger.info(f"Generated Python code saved to: {temp_file_path}")
        
        try:
            # Execute the Python code in a new context
            globals_dict = {
                'db_connector': self.db_connector,
                'model_storage_path': self.model_storage_path
            }
            
            exec(python_code, globals_dict)
            result = globals_dict.get('result', None)
            
            return result
        except Exception as e:
            logger.error(f"Error executing generated code: {e}")
            raise
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    def _generate_code(self, tree):
        """
        Generate Python code from parsed tree
        
        Args:
            tree: Parsed Lark tree
            
        Returns:
            Generated Python code as string
        """
        # Base imports for all statements
        code = [
            "import pandas as pd",
            "import numpy as np",
            "import os",
            "import pickle",
            "import json",
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
            "from sklearn.compose import ColumnTransformer",
            "from sklearn.pipeline import Pipeline",
            "from sklearn.model_selection import train_test_split",
            ""
        ]
        
        # Extract statement type and generate appropriate code
        statement_type = tree.data
        
        if statement_type == 'create_model_stmt':
            code.extend(self._generate_create_model_code(tree))
        elif statement_type == 'create_ml_view_stmt':
            code.extend(self._generate_create_ml_view_code(tree))
        elif statement_type == 'drop_model_stmt':
            code.extend(self._generate_drop_model_code(tree))
        elif statement_type == 'drop_ml_view_stmt':
            code.extend(self._generate_drop_ml_view_code(tree))
        elif statement_type == 'list_models_stmt':
            code.extend(self._generate_list_models_code())
        elif statement_type == 'model_info_stmt':
            code.extend(self._generate_model_info_code(tree))
        elif statement_type == 'evaluate_model_stmt':
            code.extend(self._generate_evaluate_model_code(tree))
        elif statement_type == 'fine_tune_model_stmt':
            code.extend(self._generate_fine_tune_model_code(tree))
        elif statement_type == 'export_model_stmt':
            code.extend(self._generate_export_model_code(tree))
        elif statement_type == 'import_model_stmt':
            code.extend(self._generate_import_model_code(tree))
        # Add other statement types as needed
        
        return "\n".join(code)
    
    def _generate_create_model_code(self, tree):
        """Generate Python code for CREATE MODEL statement"""
        model_name = tree.children[0].value
        algorithm = tree.children[1].value.strip("'\"")
        
        train_clause = tree.children[2]
        if train_clause.children[0].data == 'table_name':
            table_name = train_clause.children[0].children[0].value
            train_query = f"SELECT * FROM {table_name}"
        else:
            # SQL query provided directly
            train_query = train_clause.children[0].children[0].value.strip("'\"")
        
        # Find predict clause if it exists
        target_column = None
        for child in tree.children:
            if hasattr(child, 'data') and child.data == 'predict_clause':
                target_column = child.children[0].value
        
        # Find features clause if it exists
        features = []
        for child in tree.children:
            if hasattr(child, 'data') and child.data == 'with_features_clause':
                for feature_child in child.children:
                    features.append(feature_child.value)
        
        # Find hyperparameters if they exist
        hyperparameters = {}
        for child in tree.children:
            if hasattr(child, 'data') and child.data == 'hyperparameters_clause':
                hyperparameters = self._parse_json_object(child.children[0])
        
        # Find preprocessing options if they exist
        preprocessing = {}
        for child in tree.children:
            if hasattr(child, 'data') and child.data == 'preprocess_clause':
                preprocessing = self._parse_json_object(child.children[0])
        
        # Generate appropriate code based on algorithm
        code = []
        
        # Import needed ML library based on algorithm
        if algorithm.lower() in ['linear_regression', 'logistic_regression', 'random_forest', 'svm', 'kmeans']:
            code.append("from sklearn import linear_model, ensemble, svm, cluster")
        elif 'xgboost' in algorithm.lower():
            code.append("import xgboost as xgb")
        elif 'lightgbm' in algorithm.lower():
            code.append("import lightgbm as lgb")
        elif 'bert' in algorithm.lower() or 'gpt' in algorithm.lower() or 'transformer' in algorithm.lower():
            code.append("from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer")
            code.append("import torch")
            code.append("from datasets import Dataset")
        
        # Add code for data loading
        code.extend([
            f"# Load data from SQL",
            f"data = db_connector.execute_query('{train_query}')",
            f"",
            f"# Separate features and target",
        ])
        
        if features:
            feature_str = ", ".join([f"'{f}'" for f in features])
            code.append(f"X = data[[{feature_str}]]")
        else:
            code.append(f"# No specific features defined, using all columns except target")
            code.append(f"X = data.drop('{target_column}', axis=1) if '{target_column}' in data.columns else data")
        
        if target_column:
            code.append(f"y = data['{target_column}'] if '{target_column}' in data.columns else None")
        
        # Add preprocessing code
        if preprocessing:
            code.append(f"# Preprocessing")
            if preprocessing.get('normalize', False):
                code.append(f"numeric_features = X.select_dtypes(include=['int64', 'float64']).columns")
                code.append(f"categorical_features = X.select_dtypes(include=['object', 'category']).columns")
                
                code.append(f"preprocessor = ColumnTransformer(")
                code.append(f"    transformers=[")
                code.append(f"        ('num', StandardScaler(), numeric_features),")
                
                encode_method = preprocessing.get('encode', 'onehot')
                if encode_method == 'onehot':
                    code.append(f"        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)")
                
                code.append(f"    ],")
                code.append(f"    remainder='passthrough'")
                code.append(f")")
        
        # Add model initialization code based on algorithm
        code.append(f"# Initialize model")
        
        if algorithm.lower() == 'linear_regression':
            code.append(f"model = linear_model.LinearRegression(**{hyperparameters})")
        elif algorithm.lower() == 'logistic_regression':
            code.append(f"model = linear_model.LogisticRegression(**{hyperparameters})")
        elif algorithm.lower() == 'random_forest':
            code.append(f"model = ensemble.RandomForestClassifier(**{hyperparameters})")
        elif algorithm.lower() == 'svm':
            code.append(f"model = svm.SVC(**{hyperparameters})")
        elif algorithm.lower() == 'kmeans':
            code.append(f"model = cluster.KMeans(**{hyperparameters})")
        elif 'xgboost' in algorithm.lower():
            code.append(f"model = xgb.XGBClassifier(**{hyperparameters})")
        elif 'lightgbm' in algorithm.lower():
            code.append(f"model = lgb.LGBMClassifier(**{hyperparameters})")
        elif 'bert' in algorithm.lower() or 'transformer' in algorithm.lower():
            code.append(f"# Hugging Face model initialization")
            code.append(f"model_name = '{algorithm}'  # This should be a valid model name from Hugging Face")
            code.append(f"tokenizer = AutoTokenizer.from_pretrained(model_name)")
            code.append(f"model = AutoModelForSequenceClassification.from_pretrained(model_name, **{hyperparameters})")
        else:
            code.append(f"raise ValueError(f'Unsupported algorithm: {algorithm}')")
        
        # Create pipeline if preprocessing was specified
        if preprocessing:
            code.append(f"# Create pipeline with preprocessing")
            code.append(f"pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])")
            fit_obj = "pipeline"
        else:
            fit_obj = "model"
        
        # Add training code
        code.append(f"# Train the model")
        if target_column:
            code.append(f"{fit_obj}.fit(X, y)")
        else:  # For unsupervised algorithms like clustering
            code.append(f"{fit_obj}.fit(X)")
        
        # Save the model
        code.append(f"# Save the model")
        code.append(f"model_path = os.path.join(model_storage_path, '{model_name}')")
        code.append(f"os.makedirs(model_path, exist_ok=True)")
        
        # Save model metadata
        metadata = {
            "model_name": model_name,
            "algorithm": algorithm,
            "features": features if features else "all",
            "target_column": target_column,
            "hyperparameters": hyperparameters,
            "preprocessing": preprocessing,
            "created_at": "pd.Timestamp.now().isoformat()"
        }
        
        code.append(f"# Save model metadata")
        code.append(f"with open(os.path.join(model_path, 'metadata.json'), 'w') as f:")
        code.append(f"    json.dump({metadata}, f)")
        
        # Save the actual model
        if preprocessing:
            code.append(f"# Save the pipeline")
            code.append(f"with open(os.path.join(model_path, 'pipeline.pkl'), 'wb') as f:")
            code.append(f"    pickle.dump(pipeline, f)")
        else:
            code.append(f"# Save the model")
            code.append(f"with open(os.path.join(model_path, 'model.pkl'), 'wb') as f:")
            code.append(f"    pickle.dump(model, f)")
        
        code.append(f"print(f'Model {model_name} created successfully')")
        code.append(f"result = 'Model {model_name} created successfully'")
        
        return code
    
    def _generate_create_ml_view_code(self, tree):
        """Generate Python code for CREATE ML_VIEW statement"""
        view_name = tree.children[0].value
        select_items = tree.children[1]
        table_name = tree.children[2].value
        
        # Extract ML function if it exists
        ml_function = None
        output_column = None
        
        for item in select_items.children:
            if hasattr(item, 'data') and item.data in ('predict_function', 'classify_function', 'cluster_function', 'embedding_function', 'explain_function'):
                ml_function = item.data
                model_name = item.children[0].value
                output_column = item.children[1].value if len(item.children) > 1 else f"{ml_function.split('_')[0]}_{model_name}"
        
        code = [
            f"# Load data from table",
            f"data = db_connector.execute_query('SELECT * FROM {table_name}')",
            f"",
            f"# Load model",
            f"model_path = os.path.join(model_storage_path, '{model_name}')",
            f"",
            f"# Check if we have a pipeline or just a model",
            f"if os.path.exists(os.path.join(model_path, 'pipeline.pkl')):",
            f"    with open(os.path.join(model_path, 'pipeline.pkl'), 'rb') as f:",
            f"        model = pickle.load(f)",
            f"else:",
            f"    with open(os.path.join(model_path, 'model.pkl'), 'rb') as f:",
            f"        model = pickle.load(f)",
            f"",
            f"# Load metadata",
            f"with open(os.path.join(model_path, 'metadata.json'), 'r') as f:",
            f"    metadata = json.load(f)",
            f"",
            f"# Extract features for prediction",
            f"features = metadata['features']",
            f"if features == 'all':",
            f"    if 'target_column' in metadata and metadata['target_column'] is not None:",
            f"        X = data.drop(metadata['target_column'], axis=1) if metadata['target_column'] in data.columns else data",
            f"    else:",
            f"        X = data",
            f"else:",
            f"    X = data[features]",
            f"",
        ]
        
        # Generate code based on the ML function
        if ml_function == 'predict_function' or ml_function == 'classify_function':
            code.extend([
                f"# Apply model to make predictions",
                f"predictions = model.predict(X)",
                f"",
                f"# Add predictions to data",
                f"result_data = data.copy()",
                f"result_data['{output_column}'] = predictions",
            ])
        elif ml_function == 'cluster_function':
            code.extend([
                f"# Apply model to assign clusters",
                f"clusters = model.predict(X)",
                f"",
                f"# Add cluster assignments to data",
                f"result_data = data.copy()",
                f"result_data['{output_column}'] = clusters",
            ])
        elif ml_function == 'embedding_function':
            code.extend([
                f"# Generate embeddings",
                f"try:",
                f"    # Try using transform method (e.g., for PCA, UMAP)",
                f"    embeddings = model.transform(X)",
                f"except AttributeError:",
                f"    # For deep learning models",
                f"    from transformers import AutoTokenizer",
                f"    tokenizer = AutoTokenizer.from_pretrained(metadata.get('algorithm', 'bert-base-uncased'))",
                f"    text_column = X.columns[0]  # Assume first column is text for embedding",
                f"    encoded = tokenizer(X[text_column].tolist(), padding=True, truncation=True, return_tensors='pt')",
                f"    import torch",
                f"    with torch.no_grad():",
                f"        outputs = model(**encoded)",
                f"        embeddings = outputs.last_hidden_state[:, 0, :].numpy()",
                f"",
                f"# Add embeddings to data",
                f"result_data = data.copy()",
                f"if isinstance(embeddings, np.ndarray) and len(embeddings.shape) == 2:",
                f"    for i in range(embeddings.shape[1]):",
                f"        result_data[f'{output_column}_{i}'] = embeddings[:, i]",
                f"else:",
                f"    result_data['{output_column}'] = list(embeddings)",
            ])
        elif ml_function == 'explain_function':
            code.extend([
                f"# Generate explanations using SHAP",
                f"import shap",
                f"",
                f"explainer = shap.Explainer(model, X)",
                f"shap_values = explainer(X)",
                f"",
                f"# Add explanations to data",
                f"result_data = data.copy()",
                f"result_data['{output_column}'] = shap_values.values.tolist()",
            ])
        
        # Create the ML view in the database
        code.extend([
            f"# Create ML view in the database",
            f"db_connector.create_ml_view('{view_name}', result_data)",
            f"",
            f"print(f'ML view {view_name} created successfully')",
            f"result = 'ML view {view_name} created successfully'",
        ])
        
        return code
    
    def _generate_drop_model_code(self, tree):
        """Generate Python code for DROP MODEL statement"""
        model_name = tree.children[0].value
        
        code = [
            f"# Delete model files",
            f"import shutil",
            f"model_path = os.path.join(model_storage_path, '{model_name}')",
            f"",
            f"if os.path.exists(model_path):",
            f"    shutil.rmtree(model_path)",
            f"    print(f'Model {model_name} dropped successfully')",
            f"    result = 'Model {model_name} dropped successfully'",
            f"else:",
            f"    print(f'Model {model_name} does not exist')",
            f"    result = 'Model {model_name} does not exist'",
        ]
        
        return code
    
    def _generate_drop_ml_view_code(self, tree):
        """Generate Python code for DROP ML_VIEW statement"""
        view_name = tree.children[0].value
        
        code = [
            f"# Drop ML view from database",
            f"success = db_connector.drop_ml_view('{view_name}')",
            f"",
            f"if success:",
            f"    print(f'ML view {view_name} dropped successfully')",
            f"    result = 'ML view {view_name} dropped successfully'",
            f"else:",
            f"    print(f'ML view {view_name} does not exist')",
            f"    result = 'ML view {view_name} does not exist'",
        ]
        
        return code
    
    def _generate_list_models_code(self):
        """Generate Python code for LIST MODELS statement"""
        code = [
            f"# List all available models",
            f"models = []",
            f"",
            f"for model_name in os.listdir(model_storage_path):",
            f"    model_dir = os.path.join(model_storage_path, model_name)",
            f"    if os.path.isdir(model_dir):",
            f"        metadata_path = os.path.join(model_dir, 'metadata.json')",
            f"        if os.path.exists(metadata_path):",
            f"            with open(metadata_path, 'r') as f:",
            f"                metadata = json.load(f)",
            f"            models.append(metadata)",
            f"",
            f"# Format and return the result",
            f"result = pd.DataFrame(models)",
            f"print(result)",
        ]
        
        return code
    
    def _generate_model_info_code(self, tree):
        """Generate Python code for MODEL INFO statement"""
        model_name = tree.children[0].value
        
        code = [
            f"# Get detailed information about the model",
            f"model_path = os.path.join(model_storage_path, '{model_name}')",
            f"",
            f"if not os.path.exists(model_path):",
            f"    print(f'Model {model_name} does not exist')",
            f"    result = f'Model {model_name} does not exist'",
            f"else:",
            f"    # Load metadata",
            f"    with open(os.path.join(model_path, 'metadata.json'), 'r') as f:",
            f"        metadata = json.load(f)",
            f"    ",
            f"    # Load model to get additional info if possible",
            f"    model_file = os.path.join(model_path, 'model.pkl')",
            f"    pipeline_file = os.path.join(model_path, 'pipeline.pkl')",
            f"    ",
            f"    if os.path.exists(pipeline_file):",
            f"        with open(pipeline_file, 'rb') as f:",
            f"            obj = pickle.load(f)",
            f"        model_info = {'type': 'pipeline'} ",
            f"    elif os.path.exists(model_file):",
            f"        with open(model_file, 'rb') as f:",
            f"            obj = pickle.load(f)",
            f"        model_info = {'type': 'model'}",
            f"    else:",
            f"        obj = None",
            f"        model_info = {'type': 'unknown'}",
            f"    ",
            f"    # Add additional model information if available",
            f"    if obj is not None:",
            f"        model_info['class'] = obj.__class__.__name__",
            f"        if hasattr(obj, 'get_params'):",
            f"            model_info['parameters'] = obj.get_params()",
            f"    ",
            f"    # Combine metadata with model info",
            f"    result = {**metadata, **model_info}",
            f"    print(result)",
        ]
        
        return code
    
    def _generate_evaluate_model_code(self, tree):
        """Generate Python code for EVALUATE MODEL statement"""
        model_name = tree.children[0].value
        
        # Find test data source
        test_data_node = tree.children[1]
        if test_data_node.data == 'table_name':
            table_name = test_data_node.children[0].value
            test_query = f"SELECT * FROM {table_name}"
        else:
            # SQL query provided directly
            test_query = test_data_node.children[0].value.strip("'\"")
        
        # Find metrics if specified
        metrics = []
        for child in tree.children:
            if hasattr(child, 'data') and child.data == 'string_list':
                for metric_child in child.children:
                    metrics.append(metric_child.value.strip("'\""))
        
        code = [
            f"# Load model",
            f"model_path = os.path.join(model_storage_path, '{model_name}')",
            f"",
            f"if not os.path.exists(model_path):",
            f"    print(f'Model {model_name} does not exist')",
            f"    result = f'Model {model_name} does not exist'",
            f"else:",
            f"    # Load metadata",
            f"    with open(os.path.join(model_path, 'metadata.json'), 'r') as f:",
            f"        metadata = json.load(f)",
            f"    ",
            f"    # Load model or pipeline",
            f"    if os.path.exists(os.path.join(model_path, 'pipeline.pkl')):",
            f"        with open(os.path.join(model_path, 'pipeline.pkl'), 'rb') as f:",
            f"            model = pickle.load(f)",
            f"    else:",
            f"        with open(os.path.join(model_path, 'model.pkl'), 'rb') as f:",
            f"            model = pickle.load(f)",
            f"    ",
            f"    # Load test data",
            f"    test_data = db_connector.execute_query('{test_query}')",
            f"    ",
            f"    # Extract features and target from test data",
            f"    target_column = metadata.get('target_column')",
            f"    features = metadata.get('features')",
            f"    ",
            f"    if target_column is None:",
            f"        print('Cannot evaluate: model does not have a target column')",
            f"        result = 'Cannot evaluate: model does not have a target column'",
            f"    elif target_column not in test_data.columns:",
            f"        print(f'Cannot evaluate: target column {target_column} not found in test data')",
            f"        result = f'Cannot evaluate: target column {target_column} not found in test data'",
            f"    else:",
            f"        y_true = test_data[target_column]",
            f"        ",
            f"        if features == 'all':",
            f"            X_test = test_data.drop(target_column, axis=1) if target_column in test_data.columns else test_data",
            f"        else:",
            f"            X_test = test_data[features]",
            f"        ",
            f"        # Make predictions",
            f"        y_pred = model.predict(X_test)",
            f"        ",
            f"        # Calculate metrics",
            f"        from sklearn import metrics as sklearn_metrics",
            f"        evaluation_results = {}"
        ]
        
        if not metrics:
            # Default metrics based on problem type
            code.extend([
                f"        # Determine problem type and use appropriate metrics",
                f"        if hasattr(model, 'predict_proba'):",
                f"            # Classification metrics",
                f"            try:",
                f"                y_pred_proba = model.predict_proba(X_test)",
                f"                evaluation_results['roc_auc'] = sklearn_metrics.roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba, multi_class='ovr')",
                f"            except:",
                f"                pass",
                f"            ",
                f"            evaluation_results['accuracy'] = sklearn_metrics.accuracy_score(y_true, y_pred)",
                f"            evaluation_results['f1'] = sklearn_metrics.f1_score(y_true, y_pred, average='weighted')",
                f"            evaluation_results['precision'] = sklearn_metrics.precision_score(y_true, y_pred, average='weighted')",
                f"            evaluation_results['recall'] = sklearn_metrics.recall_score(y_true, y_pred, average='weighted')",
                f"            ",
                f"            # Confusion matrix",
                f"            cm = sklearn_metrics.confusion_matrix(y_true, y_pred)",
                f"            evaluation_results['confusion_matrix'] = cm.tolist()",
                f"        else:",
                f"            # Regression metrics",
                f"            evaluation_results['r2'] = sklearn_metrics.r2_score(y_true, y_pred)",
                f"            evaluation_results['mae'] = sklearn_metrics.mean_absolute_error(y_true, y_pred)",
                f"            evaluation_results['mse'] = sklearn_metrics.mean_squared_error(y_true, y_pred)",
                f"            evaluation_results['rmse'] = np.sqrt(evaluation_results['mse'])",
            ])
        else:
            # Use specified metrics
            code.append(f"        # Calculate specified metrics")
            
            for metric in metrics:
                if metric.lower() == 'accuracy':
                    code.append(f"        evaluation_results['accuracy'] = sklearn_metrics.accuracy_score(y_true, y_pred)")
                elif metric.lower() == 'f1':
                    code.append(f"        evaluation_results['f1'] = sklearn_metrics.f1_score(y_true, y_pred, average='weighted')")
                elif metric.lower() == 'precision':
                    code.append(f"        evaluation_results['precision'] = sklearn_metrics.precision_score(y_true, y_pred, average='weighted')")
                elif metric.lower() == 'recall':
                    code.append(f"        evaluation_results['recall'] = sklearn_metrics.recall_score(y_true, y_pred, average='weighted')")
                elif metric.lower() in ['auc', 'roc_auc']:
                    code.append(f"        try:")
                    code.append(f"            y_pred_proba = model.predict_proba(X_test)")
                    code.append(f"            evaluation_results['roc_auc'] = sklearn_metrics.roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba, multi_class='ovr')")
                    code.append(f"        except:")
                    code.append(f"            pass")
                elif metric.lower() == 'r2':
                    code.append(f"        evaluation_results['r2'] = sklearn_metrics.r2_score(y_true, y_pred)")
                elif metric.lower() in ['mae', 'mean_absolute_error']:
                    code.append(f"        evaluation_results['mae'] = sklearn_metrics.mean_absolute_error(y_true, y_pred)")
                elif metric.lower() in ['mse', 'mean_squared_error']:
                    code.append(f"        evaluation_results['mse'] = sklearn_metrics.mean_squared_error(y_true, y_pred)")
                elif metric.lower() == 'rmse':
                    code.append(f"        evaluation_results['rmse'] = np.sqrt(sklearn_metrics.mean_squared_error(y_true, y_pred))")
                elif metric.lower() == 'confusion_matrix':
                    code.append(f"        cm = sklearn_metrics.confusion_matrix(y_true, y_pred)")
                    code.append(f"        evaluation_results['confusion_matrix'] = cm.tolist()")
        
        code.extend([
            f"        ",
            f"        # Save evaluation results to model directory",
            f"        evaluation_file = os.path.join(model_path, 'evaluation.json')",
            f"        with open(evaluation_file, 'w') as f:",
            f"            json.dump(evaluation_results, f, indent=2)",
            f"        ",
            f"        print(f'Model {model_name} evaluated successfully')",
            f"        result = evaluation_results",
        ])
        
        return code
    
    def _generate_fine_tune_model_code(self, tree):
        """Generate Python code for FINE TUNE MODEL statement"""
        model_name = tree.children[0].value
        base_model_name = tree.children[1].value
        data_table = tree.children[2].value
        
        # Parse parameters if provided
        parameters = {}
        for child in tree.children:
            if hasattr(child, 'data') and child.data == 'json_object':
                parameters = self._parse_json_object(child)
        
        code = [
            f"# Import necessary libraries for fine-tuning",
            f"from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer",
            f"from datasets import Dataset",
            f"import torch",
            f"",
            f"# Load the data for fine-tuning",
            f"data = db_connector.execute_query('SELECT * FROM {data_table}')",
            f"",
            f"# Prepare directory for the fine-tuned model",
            f"model_path = os.path.join(model_storage_path, '{model_name}')",
            f"os.makedirs(model_path, exist_ok=True)",
            f"",
            f"# Determine if base model is a local model or Hugging Face model",
            f"local_base_path = os.path.join(model_storage_path, '{base_model_name}')",
            f"if os.path.exists(local_base_path):",
            f"    # Load metadata to determine model type",
            f"    with open(os.path.join(local_base_path, 'metadata.json'), 'r') as f:",
            f"        metadata = json.load(f)",
            f"    ",
            f"    if 'bert' in metadata.get('algorithm', '').lower() or 'transformer' in metadata.get('algorithm', '').lower():",
            f"        # Load from local directory",
            f"        model = AutoModelForSequenceClassification.from_pretrained(local_base_path)",
            f"        tokenizer = AutoTokenizer.from_pretrained(local_base_path)",
            f"    else:",
            f"        raise ValueError(f'Base model {base_model_name} is not a transformer model that can be fine-tuned')",
            f"else:",
            f"    # Assume it's a Hugging Face model",
            f"    try:",
            f"        model = AutoModelForSequenceClassification.from_pretrained('{base_model_name}')",
            f"        tokenizer = AutoTokenizer.from_pretrained('{base_model_name}')",
            f"    except Exception as e:",
            f"        raise ValueError(f'Error loading base model {base_model_name}: {str(e)}')",
            f"",
            f"# Prepare dataset for fine-tuning",
            f"text_column = data.columns[0]  # Assume first column is text",
            f"label_column = data.columns[1] if len(data.columns) > 1 else None  # Assume second column is label",
            f"",
            f"if label_column is None:",
            f"    raise ValueError('Need at least two columns: text and label for fine-tuning')",
            f"",
            f"# Prepare dataset",
            f"def tokenize_function(examples):",
            f"    return tokenizer(examples[text_column], padding='max_length', truncation=True)",
            f"",
            f"# Convert pandas DataFrame to Hugging Face Dataset",
            f"hf_dataset = Dataset.from_pandas(data)",
            f"tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)",
            f"",
            f"# Set up training arguments",
            f"training_args = TrainingArguments(",
            f"    output_dir=model_path,",
            f"    num_train_epochs=3,",
            f"    per_device_train_batch_size=8,",
            f"    per_device_eval_batch_size=8,",
            f"    warmup_steps=500,",
            f"    weight_decay=0.01,",
            f"    logging_dir=os.path.join(model_path, 'logs'),",
            f"    logging_steps=10,",
        ]
        
        # Add custom training parameters if provided
        for k, v in parameters.items():
            code.append(f"    {k}={v},")
        
        code.extend([
            f")",
            f"",
            f"# Create Trainer",
            f"trainer = Trainer(",
            f"    model=model,",
            f"    args=training_args,",
            f"    train_dataset=tokenized_dataset,",
            f")",
            f"",
            f"# Fine-tune the model",
            f"trainer.train()",
            f"",
            f"# Save the fine-tuned model",
            f"model.save_pretrained(model_path)",
            f"tokenizer.save_pretrained(model_path)",
            f"",
            f"# Save metadata",
            f"metadata = {",
            f"    'model_name': '{model_name}',",
            f"    'base_model': '{base_model_name}',",
            f"    'algorithm': 'fine-tuned-transformer',",
            f"    'parameters': {parameters},",
            f"    'created_at': pd.Timestamp.now().isoformat()",
            f"}",
            f"",
            f"with open(os.path.join(model_path, 'metadata.json'), 'w') as f:",
            f"    json.dump(metadata, f, indent=2)",
            f"",
            f"print(f'Model {model_name} fine-tuned successfully')",
            f"result = f'Model {model_name} fine-tuned successfully'",
        ])
        
        return code
    
    def _generate_export_model_code(self, tree):
        """Generate Python code for EXPORT MODEL statement"""
        model_name = tree.children[0].value
        export_path = tree.children[1].value.strip("'\"")
        
        code = [
            f"# Export model to the specified path",
            f"import shutil",
            f"model_path = os.path.join(model_storage_path, '{model_name}')",
            f"",
            f"if not os.path.exists(model_path):",
            f"    print(f'Model {model_name} does not exist')",
            f"    result = f'Model {model_name} does not exist'",
            f"else:",
            f"    # Create export directory if it doesn't exist",
            f"    os.makedirs(os.path.dirname('{export_path}'), exist_ok=True)",
            f"    ",
            f"    # Create a zip archive of the model directory",
            f"    shutil.make_archive('{export_path}', 'zip', model_path)",
            f"    ",
            f"    print(f'Model {model_name} exported to {export_path}.zip')",
            f"    result = f'Model {model_name} exported to {export_path}.zip'",
        ]
        
        return code
    
    def _generate_import_model_code(self, tree):
        """Generate Python code for IMPORT MODEL statement"""
        model_name = tree.children[0].value
        import_path = tree.children[1].value.strip("'\"")
        
        # Check for features specification
        features = []
        for child in tree.children:
            if hasattr(child, 'data') and child.data == 'column_list':
                for feature_child in child.children:
                    features.append(feature_child.value)
        
        code = [
            f"# Import model from the specified path",
            f"import shutil, tempfile, zipfile",
            f"model_path = os.path.join(model_storage_path, '{model_name}')",
            f"",
            f"if os.path.exists(model_path):",
            f"    print(f'Model {model_name} already exists. Please use a different name or drop the existing model.')",
            f"    result = f'Model {model_name} already exists'",
            f"else:",
            f"    # Create model directory",
            f"    os.makedirs(model_path, exist_ok=True)",
            f"    ",
            f"    # Extract the zip archive to a temporary directory",
            f"    with tempfile.TemporaryDirectory() as temp_dir:",
            f"        with zipfile.ZipFile('{import_path}.zip', 'r') as zip_ref:",
            f"            zip_ref.extractall(temp_dir)",
            f"        ",
            f"        # Copy the extracted files to the model directory",
            f"        for item in os.listdir(temp_dir):",
            f"            s = os.path.join(temp_dir, item)",
            f"            d = os.path.join(model_path, item)",
            f"            if os.path.isdir(s):",
            f"                shutil.copytree(s, d)",
            f"            else:",
            f"                shutil.copy2(s, d)",
            f"    ",
        ]
        
        # Update features if specified
        if features:
            code.extend([
                f"    # Update features in metadata",
                f"    metadata_path = os.path.join(model_path, 'metadata.json')",
                f"    if os.path.exists(metadata_path):",
                f"        with open(metadata_path, 'r') as f:",
                f"            metadata = json.load(f)",
                f"        ",
                f"        metadata['features'] = {features}",
                f"        ",
                f"        with open(metadata_path, 'w') as f:",
                f"            json.dump(metadata, f, indent=2)",
            ])
        
        code.extend([
            f"    print(f'Model {model_name} imported successfully')",
            f"    result = f'Model {model_name} imported successfully'",
        ])
        
        return code
    
    def _parse_json_object(self, json_node):
        """Parse a JSON object node from the Lark tree"""
        result = {}
        
        for pair_node in json_node.children:
            if len(pair_node.children) >= 2:
                key = pair_node.children[0].value.strip("'\"")
                value_node = pair_node.children[1]
                
                # Recursively parse the value based on its type
                if hasattr(value_node, 'data'):
                    if value_node.data == 'json_object':
                        value = self._parse_json_object(value_node)
                    elif value_node.data == 'json_array':
                        value = self._parse_json_array(value_node)
                    else:
                        value = self._parse_primitive_value(value_node)
                else:
                    value = self._parse_primitive_value(value_node)
                
                result[key] = value
        
        return result
    
    def _parse_json_array(self, array_node):
        """Parse a JSON array node from the Lark tree"""
        result = []
        
        for value_node in array_node.children:
            if hasattr(value_node, 'data'):
                if value_node.data == 'json_object':
                    value = self._parse_json_object(value_node)
                elif value_node.data == 'json_array':
                    value = self._parse_json_array(value_node)
                else:
                    value = self._parse_primitive_value(value_node)
            else:
                value = self._parse_primitive_value(value_node)
            
            result.append(value)
        
        return result
    
    def _parse_primitive_value(self, value_node):
        """Parse a primitive value node from the Lark tree"""
        if hasattr(value_node, 'type') and value_node.type == 'STRING':
            return value_node.value.strip("'\"")
        elif hasattr(value_node, 'type') and value_node.type == 'NUMBER':
            try:
                return int(value_node.value)
            except ValueError:
                return float(value_node.value)
        elif hasattr(value_node, 'value'):
            if value_node.value == 'true':
                return True
            elif value_node.value == 'false':
                return False
            elif value_node.value == 'null':
                return None
            else:
                return value_node.value
        else:
            return None


class MXQLTransformer(Transformer):
    """Transformer for Lark parse tree"""
    
    @v_args(inline=True)
    def string(self, s):
        return s[1:-1]  # Remove quotes

    @v_args(inline=True)
    def number(self, n):
        try:
            return int(n)
        except ValueError:
            return float(n)
