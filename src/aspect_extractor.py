import spacy
import numpy as np
import nltk
from nltk.corpus import opinion_lexicon
from typing import List, Dict, Tuple


class AspectSentimentAnalyzer:
    def __init__(self):
        # Load spaCy model for advanced NLP tasks
        self.nlp = spacy.load('en_core_web_sm')

        # Predefined product aspects for Electronics
        self.product_aspects = {
            'performance': ['speed', 'processing', 'efficiency', 'power', 'capability', 'fast', 'slow'],
            'battery': ['battery', 'charge', 'power consumption', 'runtime', 'charging', 'drain'],
            'design': ['build', 'design', 'size', 'weight', 'appearance', 'form factor', 'sleek', 'bulky'],
            'durability': ['quality', 'durability', 'reliability', 'robust', 'sturdy', 'break', 'fragile'],
            'features': ['feature', 'capability', 'function', 'specification', 'option', 'tool'],
            'price': ['cost', 'price', 'value', 'affordable', 'expensive', 'cheap', 'pricey'],
            'customer_support': ['support', 'service', 'help', 'assistance', 'warranty', 'customer service'],
            'compatibility': ['compatibility', 'integration', 'work with', 'connect', 'work', 'interface'],
        }

        # Sentiment lexicons with fallback
        try:
            self.positive_words = set(opinion_lexicon.words('positive-words.txt'))
            self.negative_words = set(opinion_lexicon.words('negative-words.txt'))
        except Exception:
            self.positive_words = {
                'good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful',
                'best', 'perfect', 'awesome', 'love', 'like', 'impressive'
            }
            self.negative_words = {
                'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst',
                'disappointing', 'hate', 'dislike', 'terrible'
            }

    def extract_aspects(self, text: str) -> Dict[str, List[str]]:
        """
        Extract product aspects from review text
        """
        doc = self.nlp(text.lower())

        detected_aspects = {}

        for aspect, keywords in self.product_aspects.items():
            aspect_mentions = []

            for sent in doc.sents:
                sent_text = sent.text.lower()

                matching_keywords = [kw for kw in keywords if kw in sent_text]
                if matching_keywords:
                    aspect_mentions.append({
                        'sentence': sent.text,
                        'keywords': matching_keywords
                    })

            if aspect_mentions:
                detected_aspects[aspect] = aspect_mentions

        return detected_aspects

    def aspect_sentiment_score(self, text: str, aspect: str) -> float:
        """
        Calculate sentiment score for a specific aspect
        """
        words = text.lower().split()

        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)

        total_count = positive_count + negative_count
        if total_count == 0:
            return 0

        return (positive_count - negative_count) / total_count

    def analyze_review_aspects(self, text: str) -> Dict[str, Dict]:
        """
        Comprehensive aspect sentiment analysis
        """
        aspects = self.extract_aspects(text)

        aspect_sentiments = {}
        for aspect, mentions in aspects.items():
            aspect_sentiments[aspect] = {
                'mentions': len(mentions),
                'sentiment_scores': [
                    self.aspect_sentiment_score(mention['sentence'], aspect)
                    for mention in mentions
                ],
                'avg_sentiment': np.mean([
                    self.aspect_sentiment_score(mention['sentence'], aspect)
                    for mention in mentions
                ]) if mentions else 0
            }

        return aspect_sentiments

    def prepare_aspect_features(self, reviews: List[str]) -> np.ndarray:
        """
        Prepare feature matrix based on aspect sentiments
        Ensures non-negative values for Naive Bayes
        """
        # Analyze aspects for all reviews
        aspect_features = []
        for review in reviews:
            aspects = self.analyze_review_aspects(review)

            # Create feature vector
            feature_vector = []
            for aspect in self.product_aspects.keys():
                if aspect in aspects:
                    # Map sentiment from [-1, 1] to [0, 1]
                    scaled_sentiment = (aspects[aspect]['avg_sentiment'] + 1) / 2
                    feature_vector.append(scaled_sentiment)
                else:
                    feature_vector.append(0)

            aspect_features.append(feature_vector)

        return np.array(aspect_features)

    def get_aspect_names(self) -> List[str]:
        """
        Return the list of product aspects
        """
        return list(self.product_aspects.keys())


def download_opinion_lexicon():
    """
    Download the opinion lexicon if not already available
    """
    nltk.download('opinion_lexicon')


def test_opinion_lexicon():
    """
    Test the opinion lexicon download and print sample words
    """
    from nltk.corpus import opinion_lexicon
    
    # Print first 10 positive words
    print(opinion_lexicon.words('positive-words.txt')[:10])

    # Print first 10 negative words
    print(opinion_lexicon.words('negative-words.txt')[:10])