#!/usr/bin/env python
"""
transpiler.py

This module converts MXQL scripts into executable PyCaret Python code.
It performs the following steps:
  1. Loads the MXQL grammar (mxql.lark) and parses the input MXQL script.
  2. Transforms the parse tree into an intermediate AST using MXQLTransformer.
  3. Transpiles the AST into PyCaret-compatible Python code.

Key Features:
  - Each MXQL statement converts into a set of Python statements with mandatory parts.
  - If optional elements (model parameters, tracking info, etc.) are not provided,
    default behavior is applied (e.g., using PyCaret’s best model selection, auto-tuning, etc.).
  - Highly commented code to allow users to review the generated code before execution.
  
The file supports the following MXQL statements:
  - CREATE EXPERIMENT
  - LIST EXPERIMENTS
  - EXPERIMENT INFO
  - CREATE MODEL
  - CREATE ML_VIEW
  - DROP MODEL
  - DROP ML_VIEW
  - LIST MODELS
  - MODEL INFO
  - EXPORT MODEL
  - IMPORT MODEL
  - EVALUATE MODEL
  - FINE TUNE MODEL
  - EXPLAIN MODEL
  - COMPARE MODELS
  - TUNE MODEL
  - DEPLOY MODEL
  - CREATE PIPELINE

Before executing the generated code in a live notebook, users are encouraged to review the output.
"""

import os
from lark import Lark
from transformer import MXQLTransformer

def transpile(ast):
    """
    Convert the AST (list of statements) into executable PyCaret Python code.

    Returns:
        A string containing the complete Python code.
    """
    code_lines = []
    
    # Header/comment in generated code
    code_lines.append("# Generated PyCaret Python Code")
    code_lines.append("# --------------------------------")
    code_lines.append("")
    
    # Iterate through each statement in the AST.
    for stmt in ast:
        stmt_type = stmt.get("type")
        
        # --- CREATE EXPERIMENT ---
        if stmt_type == "create_experiment":
            exp_name = stmt.get("experiment_name")
            task = stmt.get("task_type")
            data_source = stmt.get("data_source")
            predict = stmt.get("predict")
            features = stmt.get("features")
            preprocess = stmt.get("preprocess") or {}
            session_id = stmt.get("session_id") or "None"  # Fallback to None if not provided

            code_lines.append(f"# --- Create Experiment: {exp_name} ---")
            code_lines.append("from pycaret.classification import setup")
            # Assume that the training data from the source is already loaded as a DataFrame.
            code_lines.append(f"# Please ensure a DataFrame named df_{data_source} exists.")
            code_lines.append("exp_setup = setup(")
            code_lines.append(f"    data=df_{data_source},")
            code_lines.append(f"    target='{predict}',  # Target column as specified")
            # Use normalization if preprocess.normalize is set, otherwise fallback to auto-detection.
            if preprocess.get("normalize", False):
                code_lines.append("    normalize=True,  # Explicit normalization")
                code_lines.append("    normalize_method='zscore',  # Using standard scaling")
            else:
                code_lines.append("    normalize=False,  # Defaults will be applied if not specified")
            # Missing values handling
            if preprocess.get("handle_missing"):
                missing_strategy = preprocess.get("handle_missing")
                code_lines.append(f"    imputation_type='simple',")
                code_lines.append(f"    simple_imputer_strategy='{missing_strategy}',  # Strategy for missing values")
            # Session id to ensure reproducibility
            if session_id != "None":
                code_lines.append(f"    session_id='{session_id}',")
            code_lines.append("    silent=True")
            code_lines.append(")")
            code_lines.append("")

        # --- LIST EXPERIMENTS ---
        elif stmt_type == "list_experiments":
            code_lines.append("# --- List Experiments ---")
            code_lines.append("# User should implement a function to list experiments from storage")
            code_lines.append("def list_experiments(storage_path='experiments/HutchML/ML_Experiments'):")
            code_lines.append("    import os")
            code_lines.append("    experiments = os.listdir(storage_path)")
            code_lines.append("    print('Available Experiments:', experiments)")
            code_lines.append("")
            code_lines.append("list_experiments()")
            code_lines.append("")

        # --- EXPERIMENT INFO ---
        elif stmt_type == "experiment_info":
            exp_name = stmt.get("experiment_name")
            code_lines.append(f"# --- Experiment Info for: {exp_name} ---")
            code_lines.append("# Implement logic to load experiment details from file storage or S3")
            code_lines.append(f"def get_experiment_info(exp_name):")
            code_lines.append("    import pandas as pd")
            code_lines.append("    # Assuming experiments are stored as CSV")
            code_lines.append(f"    path = f'experiments/HutchML/ML_Experiments/{exp_name}/{exp_name}.csv'")
            code_lines.append("    info = pd.read_csv(path)")
            code_lines.append("    print(info)")
            code_lines.append("")
            code_lines.append(f"get_experiment_info('{exp_name}')")
            code_lines.append("")
        
        # --- CREATE MODEL ---
        elif stmt_type == "create_model":
            model_name = stmt.get("model_name")
            experiment = stmt.get("experiment")
            task = stmt.get("task_type")
            automl = stmt.get("automl")
            train_on = stmt.get("train_on")
            predict = stmt.get("predict")
            features = stmt.get("features")
            hyperparams = stmt.get("hyperparameters") or {}
            validation = stmt.get("validation") or {}
            tracking = stmt.get("tracking") or {}

            code_lines.append(f"# --- Create Model: {model_name} ---")
            code_lines.append("from pycaret.classification import create_model")
            # If no hyperparameters provided, let pycaret choose defaults.
            if hyperparams:
                params_str = ", ".join(f"'{k}': {v}" for k, v in hyperparams.items())
                code_lines.append(f"# Using custom hyperparameters: {{ {params_str} }}")
                hyperparams_code = f"custom_grid={{ {params_str} }}"
            else:
                code_lines.append("# No custom hyperparameters provided; PyCaret will use default settings")
                hyperparams_code = ""
            # AutoML flag: if AUTOML is true, additional handling might be required.
            if automl:
                code_lines.append("# Note: AUTOML flag is set. Additional AutoML logic might be implemented here.")
            code_lines.append(f"{model_name} = create_model(")
            code_lines.append("    'xgboost',  # Defaulting to xgboost; override task if needed")
            code_lines.append(f"    {hyperparams_code}")
            code_lines.append(")")
            code_lines.append("")
        
        # --- CREATE ML_VIEW ---
        elif stmt_type == "create_ml_view":
            view_name = stmt.get("view_name")
            select_items = stmt.get("select_items")
            table_name = stmt.get("table_name")
            where_clause = stmt.get("where")
            code_lines.append(f"# --- Create ML View: {view_name} ---")
            code_lines.append("from pycaret.classification import predict_model")
            code_lines.append(f"# The view is constructed using table {table_name} with select items {select_items}")
            if where_clause:
                code_lines.append(f"# WHERE clause: {where_clause}")
            code_lines.append("# Implement custom logic to create a view if needed")
            code_lines.append("")
        
        # --- DROP MODEL ---
        elif stmt_type == "drop_model":
            model_name = stmt.get("model_name")
            code_lines.append(f"# --- Drop Model: {model_name} ---")
            code_lines.append("# Use your database connection (db_connection.py) to run a DROP command for the model")
            code_lines.append(f"# Example: sql_conn.execute_command('DROP MODEL {model_name};')")
            code_lines.append("")
        
        # --- DROP ML_VIEW ---
        elif stmt_type == "drop_ml_view":
            view_name = stmt.get("view_name")
            code_lines.append(f"# --- Drop ML View: {view_name} ---")
            code_lines.append("# Use your database connection (db_connection.py) to run a DROP command for the view")
            code_lines.append(f"# Example: sql_conn.execute_command('DROP ML_VIEW {view_name};')")
            code_lines.append("")
        
        # --- LIST MODELS ---
        elif stmt_type == "list_models":
            experiment = stmt.get("experiment")
            code_lines.append("# --- List Models ---")
            if experiment:
                code_lines.append(f"# Listing models for experiment: {experiment}")
            else:
                code_lines.append("# Listing all models")
            code_lines.append("from pycaret.classification import models")
            code_lines.append("print(models())")
            code_lines.append("")
        
        # --- MODEL INFO ---
        elif stmt_type == "model_info":
            model_name = stmt.get("model_name")
            code_lines.append(f"# --- Model Info for: {model_name} ---")
            code_lines.append("from pycaret.classification import model")
            code_lines.append(f"print(model({model_name}))  # Adjust to display detailed info if available")
            code_lines.append("")
        
        # --- EXPORT MODEL ---
        elif stmt_type == "export_model":
            model_name = stmt.get("model_name")
            destination = stmt.get("destination")
            code_lines.append(f"# --- Export Model: {model_name} ---")
            code_lines.append("from pycaret.classification import save_model")
            code_lines.append(f"save_model({model_name}, '{destination}')  # Exports the model to specified destination")
            code_lines.append("")
        
        # --- IMPORT MODEL ---
        elif stmt_type == "import_model":
            model_name = stmt.get("model_name")
            source = stmt.get("source")
            code_lines.append(f"# --- Import Model: {model_name} ---")
            code_lines.append("from pycaret.classification import load_model")
            code_lines.append(f"{model_name} = load_model('{source}')  # Load model from the given source")
            code_lines.append("")
        
        # --- EVALUATE MODEL ---
        elif stmt_type == "evaluate_model":
            model_name = stmt.get("model_name")
            on_source = stmt.get("on")
            metrics = stmt.get("metrics")
            code_lines.append(f"# --- Evaluate Model: {model_name} ---")
            code_lines.append("from pycaret.classification import evaluate_model")
            code_lines.append(f"evaluate_model({model_name})  # Evaluation on data source {on_source} using metrics: {metrics}")
            code_lines.append("")
        
        # --- FINE TUNE MODEL ---
        elif stmt_type == "fine_tune_model":
            model_name = stmt.get("model_name")
            base_model = stmt.get("base_model")
            data_source = stmt.get("data")
            parameters = stmt.get("parameters")
            code_lines.append(f"# --- Fine Tune Model: {model_name} ---")
            code_lines.append("# Fine tuning might require retraining using a base model and new parameters.")
            code_lines.append("# Implement custom fine tuning logic here.")
            code_lines.append("")
        
        # --- EXPLAIN MODEL ---
        elif stmt_type == "explain_model":
            model_name = stmt.get("model_name")
            instance = stmt.get("instance")
            using = stmt.get("using")
            code_lines.append(f"# --- Explain Model: {model_name} ---")
            code_lines.append("from pycaret.classification import interpret_model")
            if instance:
                code_lines.append(f"# Explanation for instance: {instance}")
            if using:
                code_lines.append(f"# Explanation method: {using}")
            code_lines.append(f"interpret_model({model_name}, plot='summary')")
            code_lines.append("")
        
        # --- COMPARE MODELS ---
        elif stmt_type == "compare_models":
            experiment = stmt.get("experiment")
            sort_by = stmt.get("sort_by") or "AUC"
            order = stmt.get("order") or "DESC"
            top = stmt.get("top") or 0
            include = stmt.get("include") or []
            exclude = stmt.get("exclude") or []
            parameters = stmt.get("parameters") or {}
            code_lines.append(f"# --- Compare Models for experiment: {experiment} ---")
            code_lines.append("from pycaret.classification import compare_models")
            code_lines.append(f"best_models = compare_models(include={include}, sort='{sort_by}', n_select={top} )")
            code_lines.append("print(best_models)")
            code_lines.append("")
        
        # --- TUNE MODEL ---
        elif stmt_type == "tune_model":
            model_name = stmt.get("model_name")
            experiment = stmt.get("experiment")
            tuning_method = stmt.get("tuning_method") or "default"
            parameters = stmt.get("parameters") or {}
            optimize = stmt.get("optimize") or "Accuracy"
            search_grid = stmt.get("search_grid") or {}
            code_lines.append(f"# --- Tune Model: {model_name} ---")
            code_lines.append("from pycaret.classification import tune_model")
            code_lines.append(f"tuned_{model_name} = tune_model({model_name}, optimize='{optimize}', custom_grid={search_grid}, n_iter=20)")
            code_lines.append("")
        
        # --- DEPLOY MODEL ---
        elif stmt_type == "deploy_model":
            model_name = stmt.get("model_name")
            deployment_target = stmt.get("deployment_target")
            parameters = stmt.get("parameters") or {}
            code_lines.append(f"# --- Deploy Model: {model_name} ---")
            code_lines.append("from pycaret.classification import deploy_model")
            code_lines.append(f"deployed_{model_name} = deploy_model({model_name}, model_name='{model_name}',")
            code_lines.append(f"                               platform='{deployment_target}',")
            code_lines.append(f"                               authentication={parameters})")
            code_lines.append("")
        
        # --- CREATE PIPELINE ---
        elif stmt_type == "create_pipeline":
            pipeline_name = stmt.get("pipeline_name")
            experiment = stmt.get("experiment")
            steps = stmt.get("steps")
            code_lines.append(f"# --- Create Pipeline: {pipeline_name} ---")
            code_lines.append("from sklearn.pipeline import Pipeline")
            code_lines.append("from sklearn.impute import SimpleImputer")
            code_lines.append("from sklearn.preprocessing import StandardScaler")
            code_lines.append("from xgboost import XGBClassifier")
            code_lines.append("# Constructing a pipeline based on provided steps:")
            step_code = []
            for step in steps:
                s_num = step.get("step_number")
                s_name = step.get("step_name")
                s_params = step.get("parameters")
                # Map the step name to a default transformer/model if recognized;
                # otherwise, include a placeholder.
                if s_name.lower() == "impute_missing":
                    transformer_code = "('imputer', SimpleImputer(strategy='mean'))"
                elif s_name.lower() == "scale":
                    transformer_code = "('scaler', StandardScaler())"
                elif s_name.lower() == "train_model":
                    transformer_code = "('classifier', XGBClassifier(**{'n_estimators': 100}))"
                else:
                    transformer_code = f"('{s_name}', None)  # Custom parameters: {s_params}"
                step_code.append(transformer_code)
            code_lines.append(f"{pipeline_name} = Pipeline(steps=[{', '.join(step_code)}])")
            code_lines.append("print('Pipeline created: ', " + pipeline_name + ")")
            code_lines.append("")
        
        else:
            code_lines.append(f"# [Warning] Statement type '{stmt_type}' is not supported by this transpiler.")
            code_lines.append("")
    
    # Return the generated code as a single string.
    return "\n".join(code_lines)

def main():
    """
    Main routine:
      1. Loads the MXQL grammar.
      2. Parses a sample MXQL script.
      3. Transforms the parse tree into an AST.
      4. Transpiles the AST into PyCaret Python code.
      5. Prints the generated code for review before execution.
    """
    sample_mxql = """
    CREATE EXPERIMENT churn_exp
    FOR CLASSIFICATION
    ON transactions_table
    PREDICT churn
    WITH FEATURES amount AS NUMERIC, location AS CATEGORICAL
    PREPROCESS { "normalize": true, "handle_missing": "mean" }
    SESSION ID = "exp_001";

    CREATE MODEL xgb_fraud_model
    IN exp_churn_exp
    FOR CLASSIFICATION
    AUTOML
    TRAIN ON transactions_table
    PREDICT churn
    WITH FEATURES amount AS NUMERIC, location AS CATEGORICAL
    HYPERPARAMETERS { "n_estimators": 100, "max_depth": 6 }
    VALIDATION SPLIT 0.2
    TRACK WITH { "tags": ["baseline", "xgboost"] };

    COMPARE MODELS IN churn_exp
    SORT BY AUC DESC
    TOP 5
    INCLUDE "xgboost", "random_forest", "lightgbm"
    WITH PARAMETERS { "fold": 5 };

    EXPLAIN MODEL xgb_fraud_model
    FOR INSTANCE ("100.5", "New York")
    USING "shap";

    IMPORT MODEL pretrained_model
    FROM "s3://models/fraud_model.pkl";

    EXPORT MODEL xgb_fraud_model
    TO "s3://models/exported/xgb_fraud_model.pkl";

    TUNE MODEL xgb_fraud_model
    IN churn_exp
    USING "optuna"
    WITH PARAMETERS { "n_trials": 20 }
    OPTIMIZE AUC
    SEARCH GRID { "max_depth": [4,6,8], "n_estimators": [100,200] };

    DEPLOY MODEL xgb_fraud_model
    TO "mlflow"
    WITH PARAMETERS { "experiment_name": "fraud_detection", "stage": "Production" };

    CREATE PIPELINE fraud_pipeline
    IN churn_exp
    STEPS (
      (1, "impute_missing", { "strategy": "mean" }),
      (2, "scale", { "method": "standard" }),
      (3, "train_model", { "model": "xgboost", "params": { "n_estimators": 100 } })
    );
    """
    
    # Load the MXQL grammar file.
    grammar_file = "mxql.lark"
    if not os.path.exists(grammar_file):
        print(f"Error: Grammar file '{grammar_file}' not found.")
        return
    with open(grammar_file, "r") as f:
        grammar = f.read()
    
    # Parse the MXQL script.
    parser = Lark(grammar, start='start', parser='lalr', propagate_positions=True)
    try:
        parse_tree = parser.parse(sample_mxql)
    except Exception as e:
        print("Error while parsing the MXQL script:", e)
        return

    # Transform the parse tree into an AST.
    transformer = MXQLTransformer()
    ast = transformer.transform(parse_tree)
    
    # Transpile the AST into PyCaret Python code.
    generated_code = transpile(ast)
    
    # Print the generated code for the user to review.
    print("Generated PyCaret Python Code:")
    print("================================")
    print(generated_code)
    print("================================")
    print("# Review the above code carefully before executing it in your Jupyter notebook.")

if __name__ == "__main__":
    main()
