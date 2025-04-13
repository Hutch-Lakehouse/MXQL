from sqlalchemy import text
import pickle
import json

class MxqlExecutionEngine:
    def __init__(self, engine):
        self.engine = engine
        self.models = {}
        self._init_schemas()
    
    def _init_schemas(self):
        with self.engine.connect() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS ML_Models"))
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS ML_Views"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ML_Models.registry (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) UNIQUE,
                    type VARCHAR(50),
                    serialized MEDIUMBLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
    
    def store_model(self, name, model):
        serialized = pickle.dumps(model)
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO ML_Models.registry (name, type, serialized)
                VALUES (:name, :type, :serialized)
            """), {"name": name, "type": str(type(model)), "serialized": serialized})
            conn.commit()
