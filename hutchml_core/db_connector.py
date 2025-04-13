import pandas as pd
import sqlalchemy
import logging
from typing import Optional, Union, Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DBConnector")

class DBConnector:
    """
    Database connector for MXQL to interact with SQL databases
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize database connector
        
        Args:
            connection_string: SQLAlchemy connection string for the database
        """
        self.connection_string = connection_string
        self.engine = None
        self._connect()
    
    def _connect(self):
        """
        Establish connection to the database
        """
        try:
            self.engine = sqlalchemy.create_engine(self.connection_string)
            logger.info(f"Connected to database: {self.connection_string}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        
        Args:
            query: SQL query to execute
            
        Returns:
            DataFrame with query results
        """
        try:
            logger.info(f"Executing query: {query}")
            return pd.read_sql(query, self.engine)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def create_ml_view(self, view_name: str, data: pd.DataFrame) -> bool:
        """
        Create a materialized view in the database from a DataFrame
        
        Args:
            view_name: Name for the new view
            data: DataFrame with the data to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a temporary table first
            temp_table_name = f"temp_{view_name}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
            logger.info(f"Creating temporary table: {temp_table_name}")
            
            # Write DataFrame to temporary table
            data.to_sql(temp_table_name, self.engine, index=False, if_exists='replace')
            
            # Create view from temporary table
            logger.info(f"Creating ML view: {view_name}")
            with self.engine.connect() as conn:
                conn.execute(f"DROP VIEW IF EXISTS {view_name}")
                conn.execute(f"CREATE VIEW {view_name} AS SELECT * FROM {temp_table_name}")
                
                # Drop the temporary table (optional)
                # conn.execute(f"DROP TABLE {temp_table_name}")
            
            return True
        except Exception as e:
            logger.error(f"Error creating ML view: {e}")
            return False
    
    def drop_ml_view(self, view_name: str) -> bool:
        """
        Drop an ML view from the database
        
        Args:
            view_name: Name of the view to drop
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Dropping ML view: {view_name}")
            with self.engine.connect() as conn:
                conn.execute(f"DROP VIEW IF EXISTS {view_name}")
            return True
        except Exception as e:
            logger.error(f"Error dropping ML view: {e}")
            return False
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """
        Get schema information for a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with column names as keys and data types as values
        """
        try:
            # This implementation is PostgreSQL-specific
            query = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
            """
            
            schema_df = self.execute_query(query)
            return dict(zip(schema_df['column_name'], schema_df['data_type']))
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {e}")
            return {}
    
    def list_tables(self) -> List[str]:
        """
        List all tables in the database
        
        Returns:
            List of table names
        """
        try:
            # This implementation is PostgreSQL-specific
            query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            """
            
            tables_df = self.execute_query(query)
            return tables_df['table_name'].tolist()
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return []
    
    def create_federated_connection(self, connection_name: str, connection_type: str, 
                                  connection_params: Dict[str, Any]) -> bool:
        """
        Create a federated connection to an external data source
        
        Args:
            connection_name: Name for the connection
            connection_type: Type of connection (e.g., 'mysql', 'postgres', 'bigquery')
            connection_params: Parameters for the connection
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Implementation depends on federated query engine used (e.g., Hutch, Presto)
            # This is a placeholder for the actual implementation
            logger.info(f"Creating federated connection: {connection_name}")
            
            # Create connection metadata
            connection_metadata = {
                'name': connection_name,
                'type': connection_type,
                'params': connection_params
            }
            
            # Store connection metadata in the database
            pd.DataFrame([connection_metadata]).to_sql('mxql_connections', 
                                                    self.engine, 
                                                    index=False, 
                                                    if_exists='append')
            
            return True
        except Exception as e:
            logger.error(f"Error creating federated connection: {e}")
            return False
