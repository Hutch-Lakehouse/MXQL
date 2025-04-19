from lark import Lark

# Grammar snippet
mxql_grammar = r"""
start: statement+

statement: 
  | create_experiment_stmt
  | list_experiments_stmt
  | experiment_info_stmt
  | create_model_stmt
  | list_models_stmt
  | model_info_stmt
  | drop_model_stmt
  | deploy_model_stmt
  | export_model_stmt
  | import_model_stmt
  | predict_model_stmt
  | plot_metrics_stmt

create_experiment_stmt: CREATE EXPERIMENT experiment_name FOR task_type data_clause predict_clause? session_clause? experiment_tracking_clause?
list_experiments_stmt: LIST EXPERIMENTS
experiment_info_stmt: SHOW EXPERIMENT INFO experiment_name
create_model_stmt: CREATE MODEL model_name IN experiment_name automl_flag fine_tune_clause? session_clause?
list_models_stmt: LIST MODELS IN experiment_name
model_info_stmt: SHOW MODEL INFO model_name
drop_model_stmt: DROP MODEL model_name
deploy_model_stmt: DEPLOY MODEL model_name TO deployment_target
export_model_stmt: EXPORT MODEL model_name TO string_value
import_model_stmt: IMPORT MODEL model_name FROM string_value
predict_model_stmt: PREDICT MODEL model_name ON data_clause predict_store_clause?
plot_metrics_stmt: PLOT METRICS chart_type FOR target

predict_clause: PREDICT column_name
session_clause: SESSION ID "=" string_value
fine_tune_clause: FINE TUNE
predict_store_clause: AS table_name
data_clause: DATA "=" query
experiment_tracking_clause: TRACK WITH URI "=" string_value
automl_flag: AUTOML

chart_type: ROC
          | RESIDUALS
          | FEATURE_IMPORTANCE
          | CONFUSION

task_type: CLASSIFICATION | REGRESSION | CLUSTERING | ANOMALY_DETECTION | TIME_SERIES
target: experiment_name | model_name

model_name: IDENTIFIER
experiment_name: IDENTIFIER
table_name: IDENTIFIER
column_name: IDENTIFIER
deployment_target: IDENTIFIER | string_value
query: STRING
string_value: STRING

FEATURE_IMPORTANCE: "feature"i "importance"i
URI: "uri"i

CREATE: "create"i
DROP: "drop"i
LIST: "list"i
SHOW: "show"i
DEPLOY: "deploy"i
EXPORT: "export"i
IMPORT: "import"i
PREDICT: "predict"i
PLOT: "plot"i
MODEL: "model"i
MODELS: "models"i
EXPERIMENT: "experiment"i
EXPERIMENTS: "experiments"i
FOR: "for"i
ON: "on"i
WITH: "with"i
SESSION: "session"i
ID: "id"i
INFO: "info"i
IN: "in"i
TO: "to"i
TRACK: "track"i
AUTOML: "automl"i
AS: "as"i
CLASSIFICATION: "classification"i
REGRESSION: "regression"i
CLUSTERING: "clustering"i
ANOMALY_DETECTION: "anomaly_detection"i
TIME_SERIES: "time_series"i
FINE: "fine"i
TUNE: "tune"i
DATA: "data"i
METRICS: "metrics"i
CHART: "chart"i
ROC: "roc"i
RESIDUALS: "residuals"i
FEATURE: "feature"i
IMPORTANCE: "importance"i
CONFUSION: "confusion"i
SELECT: "select"i
FROM: "from"i

IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_.]+/
STRING: /"[^"]*"/ | /'[^']*'/
NUMBER: /\d+(\.\d+)?/

COMMENT: /--[^\n]*/
%import common.WS
%ignore WS
%ignore COMMENT
"""

# Create the parser using the Earley algorithm
mxql_parser = Lark(mxql_grammar, parser="earley", maybe_placeholders=True, propagate_positions=True)

def parse_mxql(query: str):
    """
    Parses an MXQL query string and returns the parse tree.
    Strips excess whitespace and gracefully handles various spacing issues.
    """
    normalized = "\n".join(line.strip() for line in query.strip().splitlines() if line.strip())
    try:
        return mxql_parser.parse(normalized)
    except Exception as e:
        raise SyntaxError(f"MXQL Parse Error: {e}") from e
