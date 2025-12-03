import requests
import pandas as pd
import json

# Example 1: JSONPlaceholder API
# url = 'https://jsonplaceholder.typicode.com/posts'
# response = requests.get(url)

# if response.status_code == 200:
#     data = response.json()
#     df = pd.DataFrame(data)
#     print(f"Extracted {len(df)} posts")
#     print(df.head())
#     df.to_csv('posts.csv', index=False)
# else:
#     print(f"Error: {response.status_code}")

# Example 2: REST Countries API
import requests
import pandas as pd

# UPDATED URL: We explicitly ask for only the fields we need
# NOTE: You cannot ask for more than 10 fields at once
url = 'https://restcountries.com/v3.1/all?fields=name,capital,population,region,flags'

response = requests.get(url)

if response.status_code == 200:
    countries = response.json()
    country_data = []
    
    # We can iterate directly now, but the structure is slightly flattened 
    # since we only requested specific fields.
    for country in countries[:20]:
        country_data.append({
            'name': country.get('name', {}).get('common', 'N/A'),
            # 'capital' is still a list
            'capital': country.get('capital', ['N/A'])[0] if country.get('capital') else 'N/A',
            'population': country.get('population', 0),
            'region': country.get('region', 'N/A')
        })
        
    df_countries = pd.DataFrame(country_data)
    df_countries.to_csv('countries.csv', index=False)
    print(df_countries)
else:
    print(f"Error: {response.status_code}")
    print(response.text) # This will print the error message if it fails again