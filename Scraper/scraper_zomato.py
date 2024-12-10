from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time
import pandas as pd
import re

class ZomatoReviewScraper:
    def __init__(self):
        # Setup Chrome options
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('--headless=new')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--window-size=1920,1080')
        self.options.add_argument('--disable-extensions')
        
        # Add user agent to avoid detection
        self.options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Disable automation detection
        self.options.add_experimental_option('excludeSwitches', ['enable-automation'])
        self.options.add_experimental_option('useAutomationExtension', False)

    def init_driver(self):
        """Initialize Chrome driver with options"""
        try:
            service = webdriver.ChromeService()
            return webdriver.Chrome(service=service, options=self.options)
        except Exception as e:
            print(f"Error initializing Chrome driver: {str(e)}")
            raise

    def wait_for_elements(self, driver, by, value, timeout=10):
        """Wait for elements to be present"""
        try:
            return WebDriverWait(driver, timeout).until(
                EC.presence_of_all_elements_located((by, value))
            )
        except TimeoutException:
            print(f"Timeout waiting for elements: {value}")
            return []

    def extract_reviews_from_page(self, html_content):
        """Extract reviews from the current page"""
        soup = BeautifulSoup(html_content, 'html.parser')
        reviews_data = []

        # Find the restaurant name
        resto_name_element = soup.find('h1', class_='sc-7kepeu-0 sc-iSDuPN fwzNdh')
        resto_name = resto_name_element.get_text().strip()

        # Find all review containers with the specific class
        review_elements = soup.find_all('p', class_='sc-1hez2tp-0 sc-hfLElm hreYiP')

        for review in review_elements:
            try:
                # Get review text
                review_text = review.text.strip()
                
                # Get timestamp if available (parent div structure)
                timestamp_element = review.find_previous('p', class_='time-stamp')
                timestamp = timestamp_element.text.strip() if timestamp_element else "No date"

                # Only add if it appears to be a review (not navigation text or other elements)
                if len(review_text) > 5 and not any(skip in review_text.lower() for skip in ['page', 'loading', 'previous', 'next']):
                    reviews_data.append({
                        'resto_name': resto_name,
                        'review_text': review_text,
                        'timestamp': timestamp
                    })
                
            except Exception as e:
                print(f"Error extracting review: {str(e)}")
                continue

        return reviews_data

    def get_next_page_url(self, current_url, page_number):
        """Generate URL for next page"""
        # Replace or add page parameter in URL
        if 'page=' in current_url:
            return re.sub(r'page=\d+', f'page={page_number}', current_url)
        else:
            separator = '&' if '?' in current_url else '?'
            return f"{current_url}{separator}page={page_number}"

    def scrape_reviews(self, urls, max_pages=10):
        """Main method to scrape reviews from multiple pages"""
        driver = self.init_driver()
        all_reviews = []
        current_page = 1

        try:
            for url in urls:
                while current_page <= max_pages:
                    page_url = self.get_next_page_url(url, current_page)
                    print(f"Scraping page {current_page}...")
                    
                    # Load the page
                    driver.get(page_url)
                    time.sleep(3)  # Wait for page to load
                    
                    # Wait for reviews to be present
                    self.wait_for_elements(driver, By.CLASS_NAME, "sc-1hez2tp-0")
                    
                    # Extract reviews from current page
                    page_reviews = self.extract_reviews_from_page(driver.page_source)
                    
                    if not page_reviews:
                        print(f"No reviews found on page {current_page}. Stopping.")
                        break
                    
                    all_reviews.extend(page_reviews)
                    print(f"Found {len(page_reviews)} reviews on page {current_page}")
                    
                    current_page += 1
                    
                    # Add delay between pages
                    time.sleep(2)
                current_page = 1

        except Exception as e:
            print(f"Error during scraping: {str(e)}")
        
        finally:
            driver.quit()

        # Convert to DataFrame
        df = pd.DataFrame(all_reviews)
        return df

if __name__ == "__main__":
    # Initialize scraper
    scraper = ZomatoReviewScraper()
    
    # URL of the Zomato reviews page
    # url = "https://www.zomato.com/id/ncr/cé-la-vie-kitchen-bar-connaught-place-new-delhi/reviews"
    urls = ["https://www.zomato.com/id/ncr/cé-la-vie-kitchen-bar-connaught-place-new-delhi/reviews", 
            "https://www.zomato.com/agra/qairo-tajganj/reviews",
            "https://www.zomato.com/varanasi/poker-mania-restaurant-bhelupur/reviews",
            "https://www.zomato.com/varanasi/pizzeria-vatika-cafe-nadesar/reviews",
            "https://www.zomato.com/varanasi/holy-chopsticks-nadesar/reviews",
            "https://www.zomato.com/the3rdfloorbse/reviews",
            "https://www.zomato.com/varanasi/de-once-more-lanka/reviews"]
    
    # Scrape reviews
    reviews_df = scraper.scrape_reviews(urls, max_pages=10)
    
    # Save to CSV
    if len(reviews_df) > 0:
        reviews_df.to_csv('zomato_reviews.csv', index=False, encoding='utf-8')
        print(f"Successfully scraped {len(reviews_df)} reviews and saved to zomato_reviews.csv")
    else:
        print("No reviews were scraped. Please check the URL and try again.")