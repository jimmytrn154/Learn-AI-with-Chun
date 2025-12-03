import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('sample.db')

# Method 1: Using pandas read_sql
query = "SELECT * FROM employees WHERE salary > 65000"
df = pd.read_sql(query, conn)
print("Employees with salary > 65000:")
print(df)

# Method 2: Using cursor
cursor = conn.cursor()
cursor.execute("SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department")
results = cursor.fetchall()
print("\nAverage salary by department:")
for row in results:
    print(f"{row[0]}: ${row[1]:.2f}")

# Method 3: Extract all data
df_all = pd.read_sql("SELECT * FROM employees", conn)
df_all.to_csv('employees.csv', index=False)
print(f"\nExtracted {len(df_all)} records to employees.csv")

conn.close()