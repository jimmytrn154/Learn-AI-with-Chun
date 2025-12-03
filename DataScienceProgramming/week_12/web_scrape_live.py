import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the website
url = 'http://quotes.toscrape.com/'

# Send GET request
response = requests.get(url)

# Check if request was successful
if response.status_code == 200:
    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all quote elements
    quotes = soup.find_all('div', class_='quote')
    
    # Extract data
    data = []
    for quote in quotes:
        text = quote.find('span', class_='text').text
        author = quote.find('small', class_='author').text
        tags = [tag.text for tag in quote.find_all('a', class_='tag')]
        data.append({
            'quote': text,
            'author': author,
            'tags': ', '.join(tags)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    print(df.head(10))
    
    # Save to CSV
    df.to_csv('quotes.csv', index=False)
    print(f"\nExtracted {len(df)} quotes and saved to quotes.csv")
else:
    print(f"Failed to retrieve page. Status code: {response.status_code}")