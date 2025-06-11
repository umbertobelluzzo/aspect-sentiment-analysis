import os
import requests
import gzip
import random
import re
import html
import pandas as pd
import json


def clean_text(text):
    """
    Normalize and clean review text

    Parameters:
    -----------
    text : str
        Input review text

    Returns:
    --------
    str
        Cleaned and normalized text
    """
    # Unescape HTML entities
    text = html.unescape(text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove excessive punctuation (keep single punctuation)
    text = re.sub(r'([.,!?])(\1+)', r'\1', text)

    return text


def process_amazon_reviews(input_file, output_file, sample_size=50000):
    """
    Process Amazon reviews dataset

    Parameters:
    -----------
    input_file : str
        Path to input gzipped JSON file
    output_file : str
        Path to output processed CSV file
    sample_size : int, optional
        Number of reviews to sample (default 50,000)

    Returns:
    --------
    pd.DataFrame
        Processed reviews dataframe
    """
    # List to store valid reviews
    processed_reviews = []

    # Set to track unique reviews (to remove duplicates)
    seen_reviews = set()

    # Open and read the gzipped file
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        # Iterate through reviews
        for line in f:
            try:
                review = json.loads(line)

                # Check for essential fields
                if not all(key in review for key in ['reviewText', 'overall']):
                    continue

                # Clean review text
                clean_review_text = clean_text(review['reviewText'])

                # Skip empty or very short reviews
                if len(clean_review_text) < 10:
                    continue

                # Remove duplicates based on review text
                if clean_review_text in seen_reviews:
                    continue
                seen_reviews.add(clean_review_text)

                # Prepare review entry
                processed_review = {
                    'review_text': clean_review_text,
                    'rating': review['overall'],
                    'sentiment': 'positive' if review['overall'] >= 4 else 'negative',
                    'verified': review.get('verified', False),
                    'review_time': review.get('reviewTime', ''),
                    'reviewer_id': review.get('reviewerID', '')
                }

                processed_reviews.append(processed_review)

                # Stop if we've reached the sample size
                if len(processed_reviews) >= sample_size:
                    break

            except json.JSONDecodeError:
                # Skip malformed entries
                continue

    # Convert to DataFrame
    df = pd.DataFrame(processed_reviews)

    # Save to CSV
    df.to_csv(output_file, index=False)

    print(f"Processed {len(df)} reviews")
    print(f"Saved to {output_file}")

    return df


def setup_data_directories():
    """
    Create necessary data directories
    """
    os.makedirs("amazon_data", exist_ok=True)

    electronics_5core_url = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
    data_dir = "amazon_data"
    electronics_file = os.path.join(data_dir, "reviews_Electronics_5.json.gz")
    
    return electronics_5core_url, data_dir, electronics_file