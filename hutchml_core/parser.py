from lark import Lark

# Define the Mxql grammar
GRAMMAR = """
start: statement+

statement: create_model_stmt
         | train_stmt
         | predict_stmt
         | evaluate_stmt
         | drop_model_stmt
         | list_models_stmt

create_model_stmt: "CREATE" "MODEL" model_name "FOR" task_type clause* ";"

train_stmt: "TRAIN" "MODEL" model_name "ON" data_source "PREDICT" column_name ";"

predict_stmt: "PREDICT" "USING" "MODEL" model_name "ON" data_source ";"

evaluate_stmt: "EVALUATE" "MODEL" model_name "ON" data_source ("WITH" "METRICS" metric_list)? ";"

drop_model_stmt: "DROP" "MODEL" model_name ";"

list_models_stmt: "LIST" "MODELS" ";"

clause: using_clause | with_features_clause | hyperparameters_clause | preprocess_clause | validation_clause

using_clause: "USING" string_value+

with_features_clause: "WITH" "FEATURES" column_list

hyperparameters_clause: "HYPERPARAMETERS" json_string

preprocess_clause: "PREPROCESS" "WITH" json_string

validation_clause: "VALIDATION" (validation_split | validation_table)

validation_split: "SPLIT" NUMBER

validation_table: "TABLE" table_name

data_source: table_name | "(" query ")"

model_name: IDENTIFIER

table_name: IDENTIFIER

column_name: IDENTIFIER

column_list: column_name ("," column_name)*

query: STRING

string_value: STRING

json_string: STRING

metric_list: string_value ("," string_value)*

task_type: "classification" | "regression" | "clustering" | "language_modeling" | "text_generation" | "image_classification" | "object_detection" | "reinforcement_learning"

IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/

STRING: /"[^"]"/ | /'[^']'/

NUMBER: /\d+(\.\d+)?/

%ignore /\s+/

%ignore /--[^\n]*/
"""

class MxqlParser:
    """A parser for the Mxql language, supporting ML/DL/AI tasks with Scikit-learn, TensorFlow, and PyTorch."""
    
    def _init_(self):
        """Initialize the parser with the Mxql grammar."""
        self.parser = Lark(GRAMMAR, start='start', parser='earley')
    
    def parse(self, code):
        """
        Parse Mxql code and return an abstract syntax tree (AST).
        
        Args:
            code (str): The Mxql code to parse.
        
        Returns:
            lark.Tree: The parsed AST.
        
        Raises:
            lark.exceptions.LarkError: If the code has syntax errors.
        """
        return self.parser.parse(code)

# Example usage
if _name_ == "_main_":
    parser = MxqlParser()
    
    # Example 1: Scikit-learn classification
    code1 = """
    CREATE MODEL my_model FOR classification
        USING "scikit-learn" "LogisticRegression"
        WITH FEATURES x1, x2
        HYPERPARAMETERS '{"C": 1.0, "penalty": "l2"}'
        VALIDATION SPLIT 0.2;
    TRAIN MODEL my_model ON my_table PREDICT y;
    """
    ast1 = parser.parse(code1)
    print("Scikit-learn example parsed successfully:")
    print(ast1.pretty())
    
    # Example 2: TensorFlow deep learning
    code2 = """
    CREATE MODEL dl_model FOR image_classification
        USING "tensorflow" "resnet50"
        HYPERPARAMETERS '{"learning_rate": 0.001, "batch_size": 32}'
        PREPROCESS WITH '{"image": {"resize": [224, 224]}}';
    PREDICT USING MODEL dl_model ON ("SELECT images FROM image_data");
    """
    ast2 = parser.parse(code2)
    print("TensorFlow example parsed successfully:")
    print(ast2.pretty())
    
    # Example 3: PyTorch language modeling
    code3 = """
    CREATE MODEL lm_model FOR language_modeling
        USING "pytorch" "transformer"
        HYPERPARAMETERS '{"num_layers": 6, "hidden_size": 512}';
    EVALUATE MODEL lm_model ON test_data WITH METRICS "perplexity";
    """
    ast3 = parser.parse(code3)
    print("PyTorch example parsed successfully:")
    print(ast3.pretty())
    
    # Example 4: Drop and list models
    code4 = """
    DROP MODEL old_model;
    LIST MODELS;
    """
    ast4 = parser.parse(code4)
    print("Management commands parsed successfully:")
    print(ast4.pretty())
