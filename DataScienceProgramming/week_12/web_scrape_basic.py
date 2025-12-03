from bs4 import BeautifulSoup
import pandas as pd

# Read HTML file
with open('sample.html', 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse HTML
soup = BeautifulSoup(html_content, 'html.parser')

# Extract table data
table = soup.find('table')
rows = table.find_all('tr')

# Extract headers
headers = [th.text.strip() for th in rows[0].find_all('th')]

# Extract data rows
data = []
for row in rows[1:]:
    cells = row.find_all('td')
    data.append([cell.text.strip() for cell in cells])

# Create DataFrame
df = pd.DataFrame(data, columns=headers)
print(df)