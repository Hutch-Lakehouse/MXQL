import json
from lark import Transformer, v_args
from typing import Dict, Any, List, Union

@v_args(inline=True)
class MxqlTransformer(Transformer):
    """
    Comprehensive transformer that converts raw parse trees into
    structured dictionaries ready for transpilation.
    Handles all MXQL statement types with context management.
    """

    def __init__(self):
        super().__init__()
        self.context = {
            'models': {},
            'views': {},
            'feature_stores': {},
            'environments': {},
            'agents': {},
            'current_scope': None
        }

    # --- Model Operations ---
    def create_model_stmt(self, create, model, name, _for, task_type, *clauses):
        model_info = {
            'type': 'model',
            'name': str(name),
            'task': str(task_type).lower(),
            'automl': False,
            'clauses': self._process_clauses(clauses)
        }
        self._register_entity('models', model_info)
        return model_info

    def train_model_stmt(self, train, model, name, on, data_source, predict, target_col):
        return {
            'type': 'train',
            'model': str(name),
            'data_source': self._process_data_source(data_source),
            'target': str(target_col)
        }

    def fine_tune_model_stmt(self, fine, tune, model, name, from_, base_model, using, *clauses):
        return {
            'type': 'fine_tune',
            'model': str(name),
            'base_model': str(base_model),
            'data': self._process_data_source(clauses[1]) if len(clauses) > 1 else None,
            'parameters': json.loads(clauses[3].value) if len(clauses) > 3 else {}
        }

    # --- Prediction/Evaluation ---
    def predict_stmt(self, predict, using, model, name, on, data_source):
        return {
            'type': 'predict',
            'model': str(name),
            'data_source': self._process_data_source(data_source)
        }

    def evaluate_model_stmt(self, evaluate, model, name, on, data_source, *metrics):
        return {
            'type': 'evaluate',
            'model': str(name),
            'data_source': self._process_data_source(data_source),
            'metrics': [str(m.value.strip('"')) for m in metrics[1:]] if metrics else ['accuracy']
        }

    def explain_model_stmt(self, explain, model, name, *args):
        explanation = {
            'type': 'explain',
            'model': str(name),
            'instance': None,
            'method': 'shap'  # default
        }
        for arg in args:
            if isinstance(arg, Tree):
                if arg.data == 'instance_clause':
                    explanation['instance'] = [str(val) for val in arg.children]
                elif arg.data == 'method_clause':
                    explanation['method'] = str(arg.children[1].value.strip('"'))
        return explanation

    # --- Feature Stores ---
    def create_feature_store_stmt(self, create, feature, store, name, from_, table, keys, *features):
        return {
            'type': 'feature_store',
            'name': str(name),
            'source_table': str(table),
            'keys': [str(col) for col in keys.children],
            'features': [self._process_feature_def(f) for f in features]
        }

    # --- RL Operations ---
    def create_environment_stmt(self, create, environment, name, type_, *params):
        return {
            'type': 'environment',
            'name': str(name),
            'env_type': str(type_.value.strip('"')),
            'parameters': json.loads(params[1].value) if params else {}
        }

    def create_agent_stmt(self, create, agent, name, for_, environment, env_name, using, model, *params):
        return {
            'type': 'agent',
            'name': str(name),
            'environment': str(env_name),
            'model': str(model.value.strip('"')),
            'hyperparameters': json.loads(params[1].value) if params else {}
        }

    # --- Utility Methods ---
    def _process_clauses(self, clauses: List[Union[Tree, Token]]) -> Dict[str, Any]:
        processed = {}
        for clause in clauses:
            if isinstance(clause, Tree):
                clause_name = clause.data.replace('_clause', '')
                if clause_name == 'using':
                    processed[clause_name] = {
                        'library': str(clause.children[0].value.strip('"')),
                        'model': str(clause.children[1].value.strip('"')) if len(clause.children) > 1 else None
                    }
                elif clause_name == 'hyperparameters':
                    processed[clause_name] = json.loads(clause.children[0].value)
                else:
                    processed[clause_name] = clause.children
        return processed

    def _process_data_source(self, source: Union[Tree, Token]]) -> Dict[str, str]:
        if isinstance(source, Token):
            return {'type': 'table', 'name': str(source)}
        else:  # Subquery
            return {'type': 'query', 'content': str(source.children[0].value)}

    def _process_feature_def(self, feature: Tree) -> Dict[str, str]:
        return {
            'name': str(feature.children[0]),
            'expression': str(feature.children[2])
        }

    def _register_entity(self, entity_type: str, entity_info: Dict[str, Any]):
        """Track created entities in context"""
        self.context[entity_type][entity_info['name']] = entity_info
        self.context['current_scope'] = entity_info['name']

    # --- Terminal Processors ---
    def STRING(self, s):
        return s.value.strip('"\'')
    
    def NUMBER(self, n):
        return float(n.value)
    
    def IDENTIFIER(self, id):
        return str(id)
    
    def json_string(self, s):
        return json.loads(s.value)
