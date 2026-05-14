# 11
# !pip install impyla
import pandas as pd
import numpy as np

class ImpalaSimulator:
    def __init__(self):
        self.data = []
        self.current_db = None

    def execute(self, query):
        print(f"DEBUG: Executing Impala SQL -> {query}")
        if "INSERT INTO" in query:
            self.data = [ {'id': 1, 'name': 'John', 'age': 25},
                          {'id': 2, 'name': 'Jane', 'age': 30},
                          {'id': 3, 'name': 'Bob', 'age': 40}
                        ]
    def fetch_as_dataframe(self):
        return pd.DataFrame(self.data)

print("[TASK 1: DATABASE & TABLE CREATION]")

cursor = ImpalaSimulator()

cursor.execute("CREATE DATABASE my_database;")

cursor.execute("USE my_database;")

cursor.execute("CREATE TABLE my_table ( id INT, name STRING, age INT );")


print("\n[TASK 2: DATA INSERTION]")

cursor.execute(
    "INSERT INTO my_table VALUES "
    "(1, 'John', 25), "
    "(2, 'Jane', 30), "
    "(3, 'Bob', 40);"
)


print("\n[TASK 3: SIMPLE QUERY & DATA ANALYSIS]")

df = cursor.fetch_as_dataframe()

print("Original Data from Impala:")

print(df)


print("\n[TASK 4: DATA PREPROCESSING & NORMALIZATION]")


def normalize_age(column):

    return (column - column.min()) / (
        column.max() - column.min()
    )


df['age_normalized'] = normalize_age(df['age'])

print("Data after Normalization (Age scaled between 0 and 1):")

print(df)


print("\n[CONCLUSION]")

print("1. Successfully simulated Impala SQL commands (Create, Insert, Select).")

print("2. Loaded data into Python environment for analysis.")

print("3. Performed Data Preprocessing using Min-Max Normalization.")