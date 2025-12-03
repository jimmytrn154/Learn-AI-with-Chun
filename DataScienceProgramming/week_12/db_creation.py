import sqlite3
import pandas as pd

# Create database and table
conn = sqlite3.connect('sample.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        department TEXT,
        salary REAL
    )
''')

# Insert sample data
employees = [
    ('Alice', 'Engineering', 75000),
    ('Bob', 'Marketing', 65000),
    ('Charlie', 'Engineering', 80000),
    ('David', 'Sales', 60000),
    ('Emma', 'Marketing', 70000)
]

cursor.executemany('INSERT INTO employees (name, department, salary) VALUES (?, ?, ?)', employees)
conn.commit()
conn.close()
print("Database created successfully!")