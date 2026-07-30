import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DB_URL = os.getenv("DB_URL", "sqlite:///data/db.sqlite")

engine = create_engine(DB_URL)
