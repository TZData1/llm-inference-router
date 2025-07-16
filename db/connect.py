# db/connect.py
import psycopg2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.manager import load_config

def get_connection():
    """Get a connection to the PostgreSQL database"""
    infra_config = load_config("infrastructure")
    db_config = infra_config.get("database", {})
    
    try:
        conn = psycopg2.connect(
            host=db_config.get("host", "localhost"),
            port=db_config.get("port", 5432),
            database=db_config.get("database", "llm_db"),
            user=db_config.get("user", "postgres"),
            password=db_config.get("password", "")
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None