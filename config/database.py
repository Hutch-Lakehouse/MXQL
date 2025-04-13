#Connection to a mysQL database for storing ML_Views,ML_Models(defaults)
from sqlalchemy import create_engine
import os

def get_engine():
    return create_engine(
        f"mysql+mysqlconnector://"
        f"{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PW')}"
        f"@{os.getenv('MYSQL_HOST')}:3306/HutchML"
    )
