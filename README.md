# OVERVIEW OF MXQL

MXQL allows you to:
Train models on data from *SQL* tables.

![image](https://github.com/user-attachments/assets/986c4c21-a9df-4bb9-854a-de2f8146361f)


Make predictions, classify data, or cluster data using trained models.
Persist results as ML views in the SQL database.
Handle preprocessing, hyperparameters, and data integration seamlessly.
It’s designed to feel familiar to SQL users, with intuitive keywords like CREATE MODEL, TRAIN ON, PREDICT, CLASSIFY, and CLUSTER. 

You typically write these statements in a SQL editor, and a transpiler converts them into Python code that runs in a background notebook using libraries like scikit-learn. The results are stored back in your SQL database as persistent ML views, which you can query just like regular SQL views.If you use an Engine like Hutch, you can be able to onnect to any datasource you have and access all that data within your sql editor without need to move data to a certain central location.

