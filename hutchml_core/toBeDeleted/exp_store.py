#!/usr/bin/env python
"""
exp_store.py

Provides a simple interface to save experiment results.
Experiments can be stored in an S3 bucket (if specified) or locally.
Naming convention:
  <Database>/ML_Experiments/<experiment_name>/<experiment_name>.<ext>
For example:
  HutchML/ML_Experiments/churn_exp/churn_exp.csv
"""

import os
import pandas as pd
import boto3

class ExperimentStorage:
    def __init__(self, s3_bucket=None, local_dir="experiments"):
        """
        Initialize storage for experiments.
        If s3_bucket is provided (e.g., "my-s3-bucket"), files will be uploaded to S3.
        Otherwise, files are saved locally under the provided local_dir.
        """
        self.s3_bucket = s3_bucket
        if s3_bucket:
            self.s3_client = boto3.client("s3")
        self.local_dir = local_dir
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

    def _get_storage_path(self, experiment_name, file_format="csv"):
        """
        Construct the storage path using a fixed database name ('HutchML') and a naming convention.
        Returns a tuple (base_path, filename).
        """
        base_path = f"HutchML/ML_Experiments/{experiment_name}"
        filename = f"{experiment_name}.{file_format}"
        return base_path, filename

    def save_experiment(self, experiment_name, df: pd.DataFrame, file_format="csv"):
        """
        Save the experiment (as CSV or Parquet) to S3 or local file system.
        """
        base_path, filename = self._get_storage_path(experiment_name, file_format)
        if self.s3_bucket:
            # Save temporarily before uploading to S3.
            temp_path = f"/tmp/{filename}"
            if file_format.lower() == "csv":
                df.to_csv(temp_path, index=False)
            elif file_format.lower() == "parquet":
                df.to_parquet(temp_path, index=False)
            else:
                raise ValueError("Unsupported file format. Use 'csv' or 'parquet'.")
            s3_key = f"{base_path}/{filename}"
            self.s3_client.upload_file(temp_path, self.s3_bucket, s3_key)
            os.remove(temp_path)
            print(f"Experiment saved to s3://{self.s3_bucket}/{s3_key}")
        else:
            # Save to local storage.
            storage_dir = os.path.join(self.local_dir, base_path)
            os.makedirs(storage_dir, exist_ok=True)
            local_path = os.path.join(storage_dir, filename)
            if file_format.lower() == "csv":
                df.to_csv(local_path, index=False)
            elif file_format.lower() == "parquet":
                df.to_parquet(local_path, index=False)
            else:
                raise ValueError("Unsupported file format. Use 'csv' or 'parquet'.")
            print(f"Experiment saved locally to {local_path}")
