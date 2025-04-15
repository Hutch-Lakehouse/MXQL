#!/usr/bin/env python
"""
db_connection.py

Contains connection classes for:
  - Trino: To query training data.
  - SQL: To connect to the database where models, views, etc. are stored.
"""

import trino
import pandas as pd
from sqlalchemy import create_engine

class TrinoConnector:
    def __init__(self, host, port, user, catalog, schema):
        self.host = host
        self.port = port
        self.user = user
        self.catalog = catalog
        self.schema = schema

    def get_connection(self):
        """Return a Trino DBAPI connection."""
        conn = trino.dbapi.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            catalog=self.catalog,
            schema=self.schema
        )
        return conn

    def query(self, sql_query):
        """
        Execute a query on Trino and return the fetched rows.
        """
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(sql_query)
        rows = cur.fetchall()
        cur.close()
        return rows

class SQLConnector:
    def __init__(self, connection_string):
        """
        Initialize with a SQLAlchemy connection string.
        For example: "sqlite:///models_storage.db" or "postgresql://user:pass@host/dbname"
        """
        self.engine = create_engine(connection_string)

    def execute(self, sql_query):
        """
        Execute a SQL query and return results as a Pandas DataFrame.
        """
        with self.engine.connect() as conn:
            df = pd.read_sql_query(sql_query, conn)
        return df

    def execute_command(self, sql_command):
        """
        Execute a SQL command that does not return rows (e.g., DDL commands).
        """
        with self.engine.connect() as conn:
            conn.execute(sql_command)
