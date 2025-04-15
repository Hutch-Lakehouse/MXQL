#!/usr/bin/env python
"""
transformer.py

This file defines MXQLTransformer, a Lark Transformer that traverses the parse tree (produced by mxql.lark)
and converts each MXQL statement into a structured AST (Abstract Syntax Tree) represented as Python dictionaries.

Each MXQL statement type from the grammar is transformed into a dictionary with a "type" field and other
related fields that capture the details of the statement.

Supported statement types:
  - create_experiment_stmt
  - list_experiments_stmt
  - experiment_info_stmt
  - create_model_stmt
  - create_ml_view_stmt
  - drop_model_stmt
  - drop_ml_view_stmt
  - list_models_stmt
  - model_info_stmt
  - export_model_stmt
  - import_model_stmt
  - evaluate_model_stmt
  - fine_tune_model_stmt
  - explain_model_stmt
  - compare_models_stmt
  - tune_model_stmt
  - deploy_model_stmt
  - create_pipeline_stmt
  
Additional rules (such as feature_spec, json_object, etc.) are also handled.
"""

from lark import Transformer

class MXQLTransformer(Transformer):

    # ------------------------------
    # Generic helper transformations
    # ------------------------------
    def __default__(self, data, children, meta):
        # If a single child exists, return it, otherwise return all children as a list.
        return children if len(children) > 1 else children[0]

    def string_value(self, items):
        s = str(items[0])
        return s.strip('"\'')
    
    def NUMBER(self, token):
        s = str(token)
        return int(s) if '.' not in s else float(s)
    
    def boolean_value(self, items):
        return True if items[0] == "true" else False

    # ------------------------------
    # Statement: CREATE EXPERIMENT
    # ------------------------------
    def create_experiment_stmt(self, items):
        # Expected order:
        # experiment_name, task_type, (table_name | query), optional predict_clause,
        # optional with_features_clause, optional preprocess clause, optional session clause.
        result = {
            "type": "create_experiment",
            "experiment_name": str(items[0]),
            "task_type": str(items[1]),
            "data_source": str(items[2]),
            "predict": None,
            "features": [],
            "preprocess": None,
            "session_id": None,
        }
        for extra in items[3:]:
            if isinstance(extra, dict):
                if extra.get("predict"):
                    result["predict"] = extra["predict"]
                if extra.get("features"):
                    result["features"] = extra["features"]
                if extra.get("preprocess"):
                    result["preprocess"] = extra["preprocess"]
                if extra.get("session_id"):
                    result["session_id"] = extra["session_id"]
        return result

    def list_experiments_stmt(self, items):
        return {"type": "list_experiments"}
    
    def experiment_info_stmt(self, items):
        return {
            "type": "experiment_info",
            "experiment_name": str(items[0])
        }
    
    # ------------------------------
    # Statement: CREATE MODEL
    # ------------------------------
    def create_model_stmt(self, items):
        # Order:
        # model_name, optional ("IN" experiment_name), task_type, optional automl_flag,
        # train_clause, optional predict_clause, optional with_features_clause, hyperparameters_clause,
        # validation_clause, optional experiment_tracking_clause.
        result = {
            "type": "create_model",
            "model_name": str(items[0]),
            "experiment": None,
            "task_type": None,
            "automl": False,
            "train_on": None,
            "predict": None,
            "features": [],
            "hyperparameters": {},
            "validation": None,
            "tracking": {}
        }
        index = 1
        # Check for optional experiment reference.
        if index < len(items) and isinstance(items[index], str) and items[index].startswith("exp_"):
            result["experiment"] = items[index][4:]  # remove potential prefix
            index += 1
        if index < len(items):
            result["task_type"] = str(items[index])
            index += 1
        if index < len(items) and items[index] == "AUTOML":
            result["automl"] = True
            index += 1
        if index < len(items):
            result["train_on"] = str(items[index])
            index += 1
        if index < len(items) and isinstance(items[index], dict) and items[index].get("predict"):
            result["predict"] = items[index]["predict"]
            index += 1
        if index < len(items) and isinstance(items[index], dict) and items[index].get("features") is not None:
            result["features"] = items[index]["features"]
            index += 1
        if index < len(items) and isinstance(items[index], dict) and items[index].get("hyperparameters") is not None:
            result["hyperparameters"] = items[index]["hyperparameters"]
            index += 1
        if index < len(items) and isinstance(items[index], dict) and items[index].get("validation") is not None:
            result["validation"] = items[index]["validation"]
            index += 1
        if index < len(items) and isinstance(items[index], dict) and items[index].get("tracking") is not None:
            result["tracking"] = items[index]["tracking"]
            index += 1
        return result

    def automl_flag(self, items):
        return "AUTOML"
    
    def train_clause(self, items):
        return str(items[0])
    
    def predict_clause(self, items):
        return {"predict": str(items[0])}
    
    def with_features_clause(self, items):
        features = []
        for feat in items:
            if isinstance(feat, dict):
                features.append(feat)
        return {"features": features}
    
    def feature_spec(self, items):
        feat = {"column": str(items[0])}
        if len(items) > 1:
            feat["type"] = str(items[1])
        else:
            feat["type"] = None
        return feat
    
    def feature_type(self, items):
        return str(items[0])
    
    def hyperparameters_clause(self, items):
        return {"hyperparameters": items[0]}
    
    def validation_clause(self, items):
        return {"validation": items[0]}
    
    def experiment_tracking_clause(self, items):
        return {"tracking": items[0]}
    
    # ------------------------------
    # Statement: CREATE ML_VIEW
    # ------------------------------
    def create_ml_view_stmt(self, items):
        # Order:
        # view_name, select_items, table_name, optional where_clause.
        result = {
            "type": "create_ml_view",
            "view_name": str(items[0]),
            "select_items": items[1],
            "table_name": str(items[2]),
            "where": None
        }
        if len(items) > 3:
            result["where"] = str(items[3])
        return result

    def select_items(self, items):
        return items

    def select_item(self, items):
        # Could be a column, *, or a ml_function.
        return items[0]  # Simplified pass-through.

    def ml_function(self, items):
        # Pass-through for functions like PREDICT, CLASSIFY, etc.
        return {"ml_function": items}

    # ------------------------------
    # Statement: DROP MODEL
    # ------------------------------
    def drop_model_stmt(self, items):
        return {"type": "drop_model", "model_name": str(items[0])}

    # ------------------------------
    # Statement: DROP ML_VIEW
    # ------------------------------
    def drop_ml_view_stmt(self, items):
        return {"type": "drop_ml_view", "view_name": str(items[0])}

    # ------------------------------
    # Statement: LIST MODELS
    # ------------------------------
    def list_models_stmt(self, items):
        result = {"type": "list_models", "experiment": None}
        if items:
            # Optional experiment specification
            result["experiment"] = str(items[0])
        return result

    # ------------------------------
    # Statement: MODEL INFO
    # ------------------------------
    def model_info_stmt(self, items):
        return {"type": "model_info", "model_name": str(items[0])}

    # ------------------------------
    # Statement: EXPORT MODEL
    # ------------------------------
    def export_model_stmt(self, items):
        return {
            "type": "export_model",
            "model_name": str(items[0]),
            "destination": self.string_value(items[1:])
        }

    # ------------------------------
    # Statement: IMPORT MODEL
    # ------------------------------
    def import_model_stmt(self, items):
        return {
            "type": "import_model",
            "model_name": str(items[0]),
            "source": self.string_value(items[1:])
        }
    
    # ------------------------------
    # Statement: EVALUATE MODEL
    # ------------------------------
    def evaluate_model_stmt(self, items):
        result = {
            "type": "evaluate_model",
            "model_name": str(items[0]),
            "on": str(items[1]),
            "metrics": None
        }
        if len(items) > 2 and isinstance(items[2], dict) and items[2].get("metrics"):
            result["metrics"] = items[2]["metrics"]
        return result

    def metric_list(self, items):
        # Return list of string values
        return [self.string_value([item]) for item in items]

    # ------------------------------
    # Statement: FINE TUNE MODEL
    # ------------------------------
    def fine_tune_model_stmt(self, items):
        result = {
            "type": "fine_tune_model",
            "model_name": str(items[0]),
            "base_model": str(items[1]),
            "data": str(items[2]),
            "parameters": {}
        }
        if len(items) > 3 and isinstance(items[3], dict):
            result["parameters"] = items[3]
        return result

    # ------------------------------
    # Statement: EXPLAIN MODEL
    # ------------------------------
    def explain_model_stmt(self, items):
        result = {
            "type": "explain_model",
            "model_name": str(items[0]),
            "instance": None,
            "using": None
        }
        for extra in items[1:]:
            if isinstance(extra, dict) and extra.get("instance"):
                result["instance"] = extra["instance"]
            elif isinstance(extra, dict) and extra.get("using"):
                result["using"] = extra["using"]
        return result

    # ------------------------------
    # Statement: COMPARE MODELS
    # ------------------------------
    def compare_models_stmt(self, items):
        result = {
            "type": "compare_models",
            "experiment": str(items[0]),
            "sort_by": None,
            "order": None,
            "top": None,
            "include": [],
            "exclude": [],
            "parameters": {}
        }
        for extra in items[1:]:
            if isinstance(extra, dict):
                if extra.get("sort_by"):
                    result["sort_by"] = extra["sort_by"]
                    result["order"] = extra.get("order")
                if extra.get("top"):
                    result["top"] = extra["top"]
                if extra.get("include"):
                    result["include"] = extra["include"]
                if extra.get("exclude"):
                    result["exclude"] = extra["exclude"]
                if extra.get("parameters"):
                    result["parameters"] = extra["parameters"]
        return result

    # ------------------------------
    # Statement: TUNE MODEL
    # ------------------------------
    def tune_model_stmt(self, items):
        result = {
            "type": "tune_model",
            "model_name": str(items[0]),
            "experiment": str(items[1]),
            "tuning_method": None,
            "parameters": {},
            "optimize": None,
            "search_grid": {}
        }
        for extra in items[2:]:
            if isinstance(extra, dict):
                if extra.get("tuning_method"):
                    result["tuning_method"] = extra["tuning_method"]
                elif extra.get("parameters"):
                    result["parameters"] = extra["parameters"]
                elif extra.get("optimize"):
                    result["optimize"] = extra["optimize"]
                elif extra.get("search_grid"):
                    result["search_grid"] = extra["search_grid"]
        return result

    # ------------------------------
    # Statement: DEPLOY MODEL
    # ------------------------------
    def deploy_model_stmt(self, items):
        result = {
            "type": "deploy_model",
            "model_name": str(items[0]),
            "deployment_target": str(items[1]),
            "parameters": {}
        }
        if len(items) > 2 and isinstance(items[2], dict):
            result["parameters"] = items[2].get("parameters", {})
        return result

    # ------------------------------
    # Statement: CREATE PIPELINE
    # ------------------------------
    def create_pipeline_stmt(self, items):
        result = {
            "type": "create_pipeline",
            "pipeline_name": str(items[0]),
            "experiment": None,
            "steps": []
        }
        idx = 1
        if idx < len(items) and isinstance(items[idx], str) and items[idx].startswith("exp_"):
            result["experiment"] = items[idx][4:]
            idx += 1
        result["steps"] = items[idx]
        return result

    def pipeline_steps(self, items):
        return items

    def pipeline_step(self, items):
        return {
            "step_number": items[0],
            "step_name": str(items[1]).strip('"'),
            "parameters": items[2]
        }
    
    # ------------------------------
    # JSON-like object definition
    # ------------------------------
    def json_object(self, items):
        result = {}
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                key, value = item
                result[key] = value
            elif isinstance(item, dict):
                result.update(item)
        return result

    def pair(self, items):
        key = self.string_value([items[0]])
        value = items[1]
        return (key, value)
    
    # ------------------------------
    # Other basic elements and rules
    # ------------------------------
    def table_name(self, items):
        return str(items[0])
    
    def column_name(self, items):
        return str(items[0])
    
    def view_name(self, items):
        return str(items[0])
    
    def model_name(self, items):
        return str(items[0])
    
    def pipeline_name(self, items):
        return str(items[0])
    
    def base_model_name(self, items):
        return str(items[0])
    
    def experiment_name(self, items):
        return str(items[0])
    
    def metric_name(self, items):
        return str(items[0])
    
    def deployment_target(self, items):
        return str(items[0])
    
    def step_number(self, items):
        return items[0]
    
    def condition(self, items):
        return str(items[0])
    
    def column_values(self, items):
        return items  # list of values

    def query(self, items):
        return self.string_value(items)
    
    def model_list(self, items):
        return [self.string_value([item]) for item in items]
    
    def metric_list(self, items):
        return [self.string_value([item]) for item in items]
    
    def tuning_method(self, items):
        return self.string_value(items)
