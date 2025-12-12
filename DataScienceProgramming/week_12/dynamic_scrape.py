from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# Set up the driver (Chrome in this example)
# If you are on a specific environment, you might need to specify the executable_path
options = webdriver.ChromeOptions()
# options.add_argument('--headless') # Uncomment to run in background
driver = webdriver.Chrome(options=options)

try:
    # We will use quotes.toscrape.com's JavaScript version
    url = 'http://quotes.toscrape.com/js/'
    print(f"Navigating to {url}...")
    driver.get(url)
    
    # Wait for the quote elements to load (dynamic content)
    # This waits up to 10 seconds for the element with class 'quote' to appear
    wait = WebDriverWait(driver, 10)
    quotes_present = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "quote")))
    
    print("Content loaded! Extracting data...")
    
    # Find all quote elements
    quote_elements = driver.find_elements(By.CLASS_NAME, "quote")
    
    data = []
    for quote in quote_elements:
        text = quote.find_element(By.CLASS_NAME, "text").text
        author = quote.find_element(By.CLASS_NAME, "author").text
        
        # Tags are also a list inside the quote
        tags_elements = quote.find_elements(By.CLASS_NAME, "tag")
        tags = [tag.text for tag in tags_elements]
        
        data.append({
            'quote': text,
            'author': author,
            'tags': ', '.join(tags)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    print("\nExtracted Data:")
    print(df.head())
    
    # Save to CSV
    df.to_csv('dynamic_quotes.csv', index=False)
    print(f"\nSaved {len(df)} quotes to dynamic_quotes.csv")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Always close the browser
    driver.quit()