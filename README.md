mxql/
│
├── grammar/
│   └── mxql.lark                      # Full DSL grammar
│
├── parser/
│   ├── __init__.py
│   ├── mxql_parser.py                # Parses DSL into Lark trees
│   ├── sql_parser.py                 # Parses SQL views using sqlglot
│
├── transformer/
│   ├── __init__.py
│   └── mxql_transformer.py           # Transforms parse trees to MXQL AST
│
├── engine/                           # Runtime Engine Core
│   ├── __init__.py
│   ├── mxql_executor.py              # Entry point: executes full MXQL scripts
│   ├── dispatcher.py                 # Dispatches each statement type to the right runner
│   ├── runners/
│   │   ├── __init__.py
│   │   ├── chain_runner.py           # Executes LangChain chains
│   │   ├── model_runner.py           # Trains or uses MLflow or PyCaret models
│   │   ├── view_runner.py            # Builds and materializes MX_VIEWs
│   │   ├── tool_runner.py            # Registers tools
│   │   ├── prompt_runner.py          # Registers prompts
│   │   └── sql_runner.py             # Handles raw SQL snippets
│   └── context.py                    # Holds execution context (space, lab, experiment, etc.)
│
├── database/                         # Postgres Bootstrapping
│   ├── init_db.py                    # Initializes all PostgreSQL tables
│   └── ddl/
│       └── schema.sql                # SQL file with all required table definitions
│
├── registry/
│   ├── prompt_registry.py
│   ├── tool_registry.py
│   ├── model_registry.py
│   └── chain_registry.py
│
├── runtime/
│   ├── artifact_manager.py          # MxqlArtifactManager
│   ├── experiment_logger.py         # Logs metadata to DB or MLflow
│   └── postgres_client.py           # For DB connection/querying
│
├── compiler/
│   ├── mxql_to_python.py            # Converts DSL to Python code for preview/debug
│   └── sqlglot_adapter.py           # Handles sqlglot-based transpilation
│
├── cli/
│   └── mxql_cli.py                  # CLI to compile, run, and debug MXQL
│
├── examples/
│   └── ...
│
├── tests/
│   └── ...
│
├── notebooks/                       # Optional: Jupyter-based experimentation
│
├── web_ui/                          # Optional: React or Streamlit frontend
│
├── .env
├── docker-compose.yml               # Spins up MXQL + Postgres
├── requirements.txt
└── README.md


# OVERVIEW OF MXQL
```
MXQL/
|
│__ ├── core/
│   │   ├── mxql_parser.py             # ✓ Complete
│   │   ├── mxql_transformer.py        # ✓ Complete (full implementation)
│   │   ├── mxql_transpiler.py         # ✓ Complete (with framework support)
│   │   └── grammar.lark               # ✓ Complete (original spec)
│   │
│   ├── execution/
│   │   ├── execution_engine.py        # ✓ Partial (needs model versioning)
│   │   └── view_manager.py            # ✓ Basic (needs dependency tracking)
│   │
│   ├── integration/
│       ├── jupyter_integration.py     # ✓ Basic (needs rich widget support)
│       └── sql_editor_plugin/         # ∅ Not started
│           └── extension.js
│______   
│       └── storage/
│       ├── model_repository.py        # ∅ Not started
│       └── view_registry.py           # ∅ Not started
│
├── database/
│   ├── init_schemas.sql               # ✓ Basic (needs full DDL)
│   └── migrations/                    # ∅ Not started
│
├── examples/
│   ├── basic_classification.ipynb     # ✓ Complete
│   ├── rl_training.ipynb              # ∅ Not started
│   └── feature_store_demo.ipynb       # ∅ Not started
│
├── tests/
│   ├── test_parser.py                 # ✓ Basic (needs full coverage)
│   ├── test_transpiler.py             # ∅ Not started
│   └── test_execution.py              # ∅ Not started
│
├── docs/
│   ├── language_spec.md               # ∅ Not started
│   └── api_reference.md               # ∅ Not started
│
├── config/
│   ├── logging.yaml                   # ∅ Not started
│   └── security_policies.yaml         # ∅ Not started
│
├── requirements.txt                   # ✓ Partial
└── setup.py                           # ∅ Not started
```
## MXQL allows you to:
- Train models on data from SQL tables.
- Make predictions, classify data, or cluster data using trained models.
- Persist results as ML views in the SQL database.
- Handle preprocessing, hyperparameters, and data integration seamlessly.
  
### It’s designed to feel familiar to SQL users, with easy natural language keywords like CREATE MODEL, TRAIN ON, PREDICT, CLASSIFY,FINE-TUNE and CLUSTER. 

You typically write these statements in a SQL editor, and a transpiler converts them into Python code that runs in a background notebook using libraries like scikit-learn. The results are stored back in your SQL database as persistent ML views, which you can query just like regular SQL views.If you use a federated  Engine like Hutch, you can be able to connect to any datasource you have and access all that data within your sql editor without need to move data to a certain central location.

```
┌─────────────────┐    ┌─────────────┐    ┌──────────────┐
│  SQL Editor     │ →  │ MXQL Parser │ →  │ Transformer  │
└─────────────────┘    └─────────────┘    └──────────────┘
        ↑                                       ↓
┌─────────────────┐    ┌─────────────┐    ┌──────────────┐
│  Results Pane   │ ←  │  Transpiler │ ←  │ Orchestrator │
└─────────────────┘    └─────────────┘    └──────────────┘
        ↑                                       ↓
┌─────────────────┐                        ┌──────────────┐
│  Jupyter Kernel │ ←───────────────────── │  DB Storage  │
└─────────────────┘                        └──────────────┘
```
## How to Use SQL Data for Machine Learning Tasks
Here’s how you can use MXQL to work with a SQL database data for training, predicting, classifying, and clustering, all within a SQL editor.

### Scenario 1: Training a Model
To train a model, you use the CREATE MODEL statement. This pulls data from a SQL table, trains a machine learning model, and saves it for later use.

### Syntax
```
CREATE MODEL <model_name>
USING algorithm = '<algorithm_name>'
TRAIN ON <table_name>
PREDICT <target_column>
[WITH FEATURES <feature1>, <feature2>, ...]
[HYPERPARAMETERS {<key>: <value>, ...}]
[PREPROCESS WITH {<option>: <value>, ...}];
```

### Example
Suppose you have a SQL table customer_data with columns age, income, tenure, and churn. You want to train a random forest model to predict churn.
```
CREATE MODEL rf_classifier
USING algorithm = 'random_forest'
TRAIN ON customer_data
PREDICT churn
WITH FEATURES age, income, tenure
HYPERPARAMETERS {n_estimators: 100, max_depth: 5}
PREPROCESS WITH {normalize: true, encode: 'onehot'};
```


Below is an example when you need to specify columns or join a few tables as the training data
```
CREATE MODEL model_name
USING algorithm = 'deep_learning_model'
TRAIN ON (SELECT * FROM table_name)
PREDICT target_column
[HYPERPARAMETERS {key: value, ...}]
PREPROCESS WITH {normalize: true, encode: 'onehot'};
```
You can replace tablename with a sql script like this:
```
TRAIN ON (SELECT a.*, b.feature 
          FROM table_a a 
          LEFT JOIN table_b b ON a.id = b.id 
          GROUP BY a.id 
          HAVING COUNT(b.id) > 1)
```
### How does it work?
The SQL editor sends this statement to the transpiler.
![image](https://github.com/user-attachments/assets/986c4c21-a9df-4bb9-854a-de2f8146361f)
The transpiler generates Python code that:

- Fetches data from customer_data using a query like SELECT age, income, tenure, churn FROM customer_data.
- Preprocesses the data (normalizes numerical columns and one-hot encodes categorical ones, if applicable).
- Trains a RandomForestClassifier with n_estimators=100 and max_depth=5.
- Saves the trained model as rf_classifier for future use.

### Scenario 2: Making Predictions
To predict on new data, use the CREATE ML_VIEW statement with the PREDICT function. This applies the trained model to a SQL table and persists the results as an ML view.
### Syntax:
```
CREATE ML_VIEW <view_name> AS
SELECT *, PREDICT(<model_name>) AS <result_column>
FROM <table_name>;
```

### Example
Let's say, you have a table new_customers with the same columns (age, income, tenure). You want to predict churn using rf_classifier.
```
CREATE ML_VIEW churn_predictions AS
SELECT *, PREDICT(rf_classifier) AS predicted_churn
FROM new_customers;
```

### How does it work?
The transpiler generates Python code that:

- Loads the saved rf_classifier model.
- Fetches data from new_customers with SELECT age, income, tenure FROM new_customers.
- Applies the model to predict churn for each row.
- Creates a persistent ML view churn_predictions in the SQL database, adding the predicted_churn column alongside the original data.

You can then query this view in your SQL editor: SELECT * FROM churn_predictions.

## Scenario 3: Classification
For classification tasks (e.g., binary or multi-class), use the CLASSIFY function in an ML view. The process is similar to prediction but explicitly indicates a classification task.
Syntax:
```
CREATE ML_VIEW <view_name> AS
SELECT *, CLASSIFY(<model_name>) AS <result_column>
FROM <table_name>;
```
### Example
Train a logistic regression model on the iris_data table (columns: sepal_length, sepal_width, petal_length, petal_width, species) and classify new data.
```
CREATE MODEL log_reg_classifier
USING algorithm = 'logistic_regression'
TRAIN ON iris_data
PREDICT species
WITH FEATURES sepal_length, sepal_width, petal_length, petal_width;

CREATE ML_VIEW iris_classifications AS
SELECT *, CLASSIFY(log_reg_classifier) AS predicted_species
FROM iris_test_data;
```

### How it works?
The first statement above trains a logistic regression model and saves it as log_reg_classifier.

The second statement:

- Loads log_reg_classifier.
- Fetches data from iris_test_data.
- Classifies each row and stores the results in the iris_classifications ML view with a predicted_species column.
  
You can Query it with: SELECT * FROM iris_classifications.

## Scenario 4: Clustering
For clustering, train a clustering model and use the CLUSTER function to assign cluster IDs to data.
Syntax:
```
CREATE MODEL <model_name>
USING algorithm = '<clustering_algorithm>'
TRAIN ON <table_name>
[WITH FEATURES <feature1>, <feature2>, ...]
[HYPERPARAMETERS {<key>: <value>, ...}];

CREATE ML_VIEW <view_name> AS
SELECT *, CLUSTER(<model_name>) AS <result_column>
FROM <table_name>;
```
### Example
Cluster customers in the sales_data table (columns: purchase_amount, frequency) into 3 groups using KMeans.
```
CREATE MODEL kmeans_model
USING algorithm = 'kmeans'
TRAIN ON sales_data
WITH FEATURES purchase_amount, frequency
HYPERPARAMETERS {n_clusters: 3};

CREATE ML_VIEW customer_segments AS
SELECT *, CLUSTER(kmeans_model) AS segment
FROM sales_data;
```
### How it works
The first statement trains a KMeans model with 3 clusters and saves it as kmeans_model.
The second statement:
- Loads kmeans_model.
- Fetches data from sales_data.
- Assigns cluster IDs to each row and creates the customer_segments ML view with a segment column.
- Query it with: SELECT * FROM customer_segments.

## Fine-Tuning Models

```
-- Create and train a deep learning model
CREATE MODEL dl_model
USING algorithm = 'deep_learning_model'
TRAIN ON (SELECT a.*, b.feature 
          FROM training_data a 
          LEFT JOIN features b ON a.id = b.id 
          GROUP BY a.id 
          HAVING COUNT(b.id) > 0)
PREDICT target
HYPERPARAMETERS {learning_rate: 0.001, epochs: 50};

-- Fine-tune the model
FINE-TUNE dl_model
ON (SELECT a.*, b.feature 
    FROM new_data a 
    LEFT JOIN features b ON a.id = b.id)
WITH HYPERPARAMETERS {learning_rate: 0.0001};

-- Make predictions and persist as a view
CREATE ML_VIEW predictions_view AS
SELECT *, PREDICT(dl_model) AS predicted_target
FROM (SELECT a.*, b.feature 
      FROM test_data a 
      LEFT JOIN features b ON a.id = b.id);
```
# Integration with SQL Editor and Transpiler

Here’s how the process works end-to-end:

### Write in SQL Editor
You write DSL statements (e.g., CREATE MODEL, CREATE ML_VIEW) in your SQL editor (e.g., pgAdmin, DBeaver).
### Transpilation:
A transpiler, linked to the SQL editor, parses the DSL using a Lark grammar (defined below).
It converts each statement into Python code that:
Connects to your SQL database (e.g., via pandas.read_sql).
Fetches the specified table data.
Executes the machine learning task using libraries like scikit-learn.
Saves models and persists ML views back to the database.

### Background Notebook
The Python code runs in a background Jupyter notebook or similar environment.
It handles data fetching, training, prediction/classification/clustering, and result storage.

### Persistent ML Views
Results are stored as ML views in the SQL database (e.g., as tables or materialized views).
These views persist until dropped, just like SQL views, and can be queried anytime.
