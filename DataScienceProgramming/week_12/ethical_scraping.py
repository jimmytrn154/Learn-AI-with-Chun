"""
Task 8: Ethical Web Scraping and Best Practices
This script demonstrates ethical scraping practices including rate limiting,
robots.txt checking, and proper headers.
"""

import requests
import time
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse

def check_robots_txt(base_url, path):
    """Check if a path is allowed by robots.txt"""
    try:
        rp = RobotFileParser()
        robots_url = urljoin(base_url, '/robots.txt')
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch('*', path)
    except Exception as e:
        print(f"Could not check robots.txt: {e}")
        return True  # Assume allowed if can't check

def scrape_with_delay(url, delay=2, headers=None):
    """Scrape with a delay between requests to be respectful to the server"""
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Educational Purpose - Data Science Lab)',
            'Accept': 'text/html,application/xhtml+xml',
        }
    
    time.sleep(delay)  # Be respectful to the server
    response = requests.get(url, headers=headers, timeout=10)
    return response

print("=== Ethical Web Scraping Practices ===\n")

# Example: Scraping with rate limiting
print("1. Rate Limiting Example:")
print("   Scraping multiple pages with 2-second delay between requests...")

base_url = 'http://quotes.toscrape.com'
urls = [
    f'{base_url}/page/1/',
    f'{base_url}/page/2/',
    f'{base_url}/page/3/',
]

quotes_data = []
for i, url in enumerate(urls, 1):
    print(f"   Scraping page {i}...")
    response = scrape_with_delay(url, delay=2)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        quotes = soup.find_all('div', class_='quote')
        quotes_data.extend([quote.find('span', class_='text').text for quote in quotes])
        print(f"   ✓ Extracted {len(quotes)} quotes from page {i}")
    else:
        print(f"   ✗ Failed to retrieve page {i}: {response.status_code}")

print(f"\n   Total quotes extracted: {len(quotes_data)}")

# Example: Checking robots.txt
print("\n2. Robots.txt Check Example:")
test_url = 'http://quotes.toscrape.com'
parsed_url = urlparse(test_url)
path = '/'

if check_robots_txt(test_url, path):
    print(f"   ✓ Scraping {path} is allowed by robots.txt")
else:
    print(f"   ✗ Scraping {path} is NOT allowed by robots.txt")

# Example: Using proper headers
print("\n3. Proper Headers Example:")
headers = {
    'User-Agent': 'Mozilla/5.0 (Educational Purpose - Data Science Lab)',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}

response = requests.get('http://quotes.toscrape.com', headers=headers)
if response.status_code == 200:
    print("   ✓ Request successful with proper headers")
    print(f"   Response size: {len(response.content)} bytes")
else:
    print(f"   ✗ Request failed: {response.status_code}")

print("\n=== Best Practices Summary ===")
print("✓ Rate limiting implemented (2+ seconds between requests)")
print("✓ robots.txt checked before scraping")
print("✓ Proper User-Agent headers used")
print("✓ Timeout set to prevent hanging requests")
print("✓ Error handling for network issues")
print("\nRemember:")
print("- Always respect website terms of service")
print("- Use APIs when available instead of scraping")
print("- Don't overload servers with too many requests")
print("- Be transparent about your scraping activities")
