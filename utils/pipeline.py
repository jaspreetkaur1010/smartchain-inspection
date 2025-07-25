import sqlite3
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def ingest_batch(df, db_path="db/supplychain.db"):
    # Encode Inspection results if not already numeric
    if df["Inspection results"].dtype == object:
        le = LabelEncoder()
        df["Inspection results"] = le.fit_transform(df["Inspection results"])
    conn = sqlite3.connect(db_path)
    df.sample(10).to_sql("supply_data", conn, if_exists='append', index=False)
    conn.close() 