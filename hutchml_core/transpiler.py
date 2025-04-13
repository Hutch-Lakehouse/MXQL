import json
from lark import Tree, Token

class MxqlTranspiler:
    def _init_(self):
        self.models = {}  # To store model instances (simulated here as a dictionary)

    def transpile(self, tree):
        """Main transpilation method that dispatches based on statement type."""
        statement_type = tree.data
        if statement_type == 'create_model_stmt':
            return self.transpile_create_model(tree)
        elif statement_type == 'train_model_stmt':
            return self.transpile_train_model(tree)
        elif statement_type == 'predict_stmt':
            return self.transpile_predict(tree)
        elif statement_type == 'evaluate_stmt':
            return self.transpile_evaluate(tree)
        elif statement_type == 'drop_model_stmt':
            return self.transpile_drop_model(tree)
        elif statement_type == 'list_models_stmt':
            return self.transpile_list_models(tree)
        elif statement_type == 'fine_tune_model_stmt':
            return self.transpile_fine_tune_model(tree)
        else:
            raise ValueError(f"Unsupported statement type: {statement_type}")

    def extract_create_model_details(self, tree):
        """Extract details from CREATE MODEL statement."""
        model_name = tree.children[2].value  # e.g., "my_model"
        task_type = tree.children[4].value  # e.g., "classification"
        library = None
        specific_model = None
        hyperparameters = {}
        for child in tree.children[5:]:
            if isinstance(child, Tree) and child.data == 'using_clause':
                library = child.children[1].value.strip('"')  # e.g., "scikit-learn"
                if len(child.children) > 2:
                    specific_model = child.children[2].value.strip('"')  # e.g., "LogisticRegression"
            elif isinstance(child, Tree) and child.data == 'hyperparameters_clause':
                hp_str = child.children[1].value
                hyperparameters = json.loads(hp_str)
        return model_name, task_type, library, specific_model, hyperparameters

    def transpile_create_model(self, tree):
        """Transpile CREATE MODEL statement into Python code."""
        model_name, task_type, library, specific_model, hyperparameters = self.extract_create_model_details(tree)
        code = ""
        
        if library == "automl":
            code = f"""
from pycaret.{task_type} import *

def train_{model_name}(query, target):
    df = pd.read_sql(query, con=engine)
    setup(data=df, target=target, silent=True)
    best_model = compare_models()
    tuned_model = tune_model(best_model)
    models['{model_name}'] = tuned_model

def predict_{model_name}(query):
    df = pd.read_sql(query, con=engine)
    predictions = predict_model(models['{model_name}'], data=df)
    return predictions['Label'] if '{task_type}' == 'classification' else predictions['prediction']

def evaluate_{model_name}(query, metrics):
    df = pd.read_sql(query, con=engine)
    predictions = predict_model(models['{model_name}'], data=df)
    y_true = df[target]
    y_pred = predictions['Label'] if '{task_type}' == 'classification' else predictions['prediction']
    for metric in metrics:
        if metric == 'accuracy':
            from sklearn.metrics import accuracy_score
            score = accuracy_score(y_true, y_pred)
            print(f"Accuracy: {{score}}")
"""
        elif library == "scikit-learn":
            specific_model = specific_model or "LogisticRegression"  # Default model
            code = f"""
from sklearn.{task_type} import {specific_model}
from sklearn.metrics import accuracy_score

models['{model_name}'] = {specific_model}(**{hyperparameters})

def train_{model_name}(query, target):
    df = pd.read_sql(query, con=engine)
    X = df.drop(columns=[target])
    y = df[target]
    models['{model_name}'].fit(X, y)

def predict_{model_name}(query):
    df = pd.read_sql(query, con=engine)
    predictions = models['{model_name}'].predict(df)
    return predictions

def evaluate_{model_name}(query, metrics):
    df = pd.read_sql(query, con=engine)
    X = df.drop(columns=[target])
    y_true = df[target]
    y_pred = models['{model_name}'].predict(X)
    for metric in metrics:
        if metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
            print(f"Accuracy: {{score}}")
"""
        elif library == "tensorflow":
            specific_model = specific_model or "Sequential"  # Simplified
            code = f"""
import tensorflow as tf

models['{model_name}'] = tf.keras.{specific_model}([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(None,)),
    tf.keras.layers.Dense(1 if '{task_type}' == 'regression' else 2, activation='sigmoid')
])

def train_{model_name}(query, target):
    df = pd.read_sql(query, con=engine)
    X = df.drop(columns=[target]).values
    y = df[target].values
    models['{model_name}'].compile(optimizer='adam', loss='binary_crossentropy' if '{task_type}' == 'classification' else 'mse')
    models['{model_name}'].fit(X, y, epochs=10, batch_size=32)

def predict_{model_name}(query):
    df = pd.read_sql(query, con=engine)
    predictions = models['{model_name}'].predict(df.values)
    return predictions.flatten()

def evaluate_{model_name}(query, metrics):
    df = pd.read_sql(query, con=engine)
    X = df.drop(columns=[target]).values
    y_true = df[target].values
    y_pred = models['{model_name}'].predict(X).flatten()
    for metric in metrics:
        if metric == 'accuracy':
            from sklearn.metrics import accuracy_score
            score = accuracy_score(y_true, y_pred.round())
            print(f"Accuracy: {{score}}")
"""
        elif library == "pytorch":
            code = f"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def _init_(self):
        super(Model, self)._init_()
        self.layer = nn.Linear(10, 1 if '{task_type}' == 'regression' else 2)  # Simplified
    def forward(self, x):
        return self.layer(x)

models['{model_name}'] = Model()

def train_{model_name}(query, target):
    df = pd.read_sql(query, con=engine)
    X = torch.tensor(df.drop(columns=[target]).values, dtype=torch.float32)
    y = torch.tensor(df[target].values, dtype=torch.float32)
    optimizer = torch.optim.Adam(models['{model_name}'].parameters())
    criterion = nn.MSELoss() if '{task_type}' == 'regression' else nn.CrossEntropyLoss()
    for _ in range(10):  # epochs
        optimizer.zero_grad()
        outputs = models['{model_name}'](X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

def predict_{model_name}(query):
    df = pd.read_sql(query, con=engine)
    X = torch.tensor(df.values, dtype=torch.float32)
    with torch.no_grad():
        predictions = models['{model_name}'](X)
    return predictions.numpy()

def evaluate_{model_name}(query, metrics):
    df = pd.read_sql(query, con=engine)
    X = torch.tensor(df.drop(columns=[target]).values, dtype=torch.float32)
    y_true = df[target].values
    with torch.no_grad():
        y_pred = models['{model_name}'](X).numpy()
    for metric in metrics:
        if metric == 'accuracy':
            from sklearn.metrics import accuracy_score
            score = accuracy_score(y_true, y_pred.round())
            print(f"Accuracy: {{score}}")
"""
        elif library == "huggingface":
            specific_model = specific_model or "bert-base-uncased"
            code = f"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained('{specific_model}')
model = AutoModelForSequenceClassification.from_pretrained('{specific_model}')
models['{model_name}'] = model
tokenizers['{model_name}'] = tokenizer

def train_{model_name}(query, target):
    from datasets import Dataset
    df = pd.read_sql(query, con=engine)
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(
        lambda x: tokenizers['{model_name}'](x['text'], padding='max_length', truncation=True),
        batched=True
    )
    training_args = TrainingArguments(
        output_dir='./results_{model_name}',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=models['{model_name}'],
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()

def predict_{model_name}(query):
    from datasets import Dataset
    df = pd.read_sql(query, con=engine)
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(
        lambda x: tokenizers['{model_name}'](x['text'], padding='max_length', truncation=True),
        batched=True
    )
    trainer = Trainer(model=models['{model_name}'])
    predictions = trainer.predict(tokenized_dataset).predictions
    return predictions

def evaluate_{model_name}(query, metrics):
    from datasets import Dataset
    df = pd.read_sql(query, con=engine)
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(
        lambda x: tokenizers['{model_name}'](x['text'], padding='max_length', truncation=True),
        batched=True
    )
    trainer = Trainer(model=models['{model_name}'])
    eval_results = trainer.evaluate(tokenized_dataset)
    for metric in metrics:
        if metric in eval_results:
            print(f"{{metric}}: {{eval_results[metric]}}")
"""
        return code.strip()

    def get_query_from_data_source(self, data_source_node):
        """Extract query string from data_source (table name or subquery)."""
        if isinstance(data_source_node, Token):
            return f"SELECT * FROM {data_source_node.value}"
        else:  # Assuming subquery is a Tree with query as first child
            return data_source_node.children[0].value

    def transpile_train_model(self, tree):
        """Transpile TRAIN MODEL statement."""
        model_name = tree.children[2].value
        data_source_node = tree.children[4]
        query = self.get_query_from_data_source(data_source_node)
        target = tree.children[6].value
        return f"train_{model_name}('{query}', '{target}')"

    def transpile_predict(self, tree):
        """Transpile PREDICT statement."""
        model_name = tree.children[3].value
        data_source_node = tree.children[5]
        query = self.get_query_from_data_source(data_source_node)
        return f"predictions = predict_{model_name}('{query}')"

    def transpile_evaluate(self, tree):
        """Transpile EVALUATE statement."""
        model_name = tree.children[2].value
        data_source_node = tree.children[4]
        query = self.get_query_from_data_source(data_source_node)
        metrics = [child.value.strip('"') for child in tree.children[6:]] if len(tree.children) > 6 else ['accuracy']
        return f"evaluate_{model_name}('{query}', {metrics})"

    def transpile_drop_model(self, tree):
        """Transpile DROP MODEL statement."""
        model_name = tree.children[2].value
        return f"del models['{model_name}']"

    def transpile_list_models(self, tree):
        """Transpile LIST MODELS statement."""
        return "print(list(models.keys()))"

    def transpile_fine_tune_model(self, tree):
        """Transpile FINE TUNE MODEL statement."""
        model_name = tree.children[2].value
        data_source_node = tree.children[4]
        query = self.get_query_from_data_source(data_source_node)
        # Assuming target is specified or defaulted; adjust based on grammar
        target = tree.children[6].value if len(tree.children) > 6 else 'labels'
        # For simplicity, reuse train function, assuming it handles fine-tuning for Hugging Face
        return f"train_{model_name}('{query}', '{target}')"

# Example usage (assuming a parsed tree is provided)
if _name_ == "_main_":
    import pandas as pd
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///example.db')
    models = {}
    tokenizers = {}  # For Hugging Face
    transpiler = MxqlTranspiler()
    # Simulated tree; replace with actual Lark parser output
    from lark import Tree, Token
    sample_tree = Tree('create_model_stmt', [
        Token('CREATE', 'CREATE'), Token('MODEL', 'MODEL'), Token('NAME', 'my_model'),
        Token('FOR', 'FOR'), Token('TASK', 'classification'),
        Tree('using_clause', [Token('USING', 'USING'), Token('STRING', '"scikit-learn"'), Token('STRING', '"LogisticRegression"')])
    ])
    print(transpiler.transpile(sample_tree))
