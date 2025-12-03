import pandas as pd
import json
import xml.etree.ElementTree as ET
from PyPDF2 import PdfReader

# 1. Extract from CSV
print("=== CSV Extraction ===")
df_csv = pd.read_csv('data.csv')
print(df_csv.head())

# 2. Extract from JSON
print("\n=== JSON Extraction ===")
with open('data.json', 'r') as file:
    json_data = json.load(file)
df_json = pd.DataFrame(json_data)
print(df_json.head())

# 3. Extract from XML
print("\n=== XML Extraction ===")
tree = ET.parse('data.xml')
root = tree.getroot()

xml_data = []
for item in root.findall('item'):
    xml_data.append({
        'name': item.find('name').text,
        'value': item.find('value').text
    })
df_xml = pd.DataFrame(xml_data)
print(df_xml)

# 4. Extract text from PDF (basic)
print("\n=== PDF Text Extraction ===")
try:
    reader = PdfReader('sample.pdf')
    text = ""
    for page in reader.pages[:3]:  # First 3 pages
        text += page.extract_text()
    print(f"Extracted {len(text)} characters from PDF")
    print(text[:500])  # First 500 characters
except Exception as e:
    print(f"PDF extraction error: {e}")