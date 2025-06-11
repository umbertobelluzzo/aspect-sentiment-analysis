import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from .aspect_extractor import AspectSentimentAnalyzer


class FeatureEngineer:
    def __init__(self, max_features=5000):
        """
        Initialize feature engineering pipeline

        Parameters:
        -----------
        max_features : int, optional (default=5000)
            Maximum number of TF-IDF features to use
        """
        # Initialize aspect sentiment analyzer
        self.aspect_analyzer = AspectSentimentAnalyzer()

        # TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )

        # MinMax Scaler for aspect features (ensures [0, 1] range)
        self.aspect_scaler = MinMaxScaler()

    def prepare_features(self, reviews):
        """
        Prepare hybrid features combining TF-IDF and aspect sentiments

        Parameters:
        -----------
        reviews : list of str
            Raw review texts

        Returns:
        --------
        scipy.sparse matrix
            Combined feature matrix
        """
        # Extract aspect sentiment features
        aspect_features = self.aspect_analyzer.prepare_aspect_features(reviews)

        # Scale aspect features to [0, 1] range
        scaled_aspect_features = self.aspect_scaler.fit_transform(aspect_features)

        # TF-IDF vectorization
        tfidf_features = self.tfidf_vectorizer.fit_transform(reviews)

        # Combine features
        # Use hstack to combine sparse (TF-IDF) and dense (aspect) features
        combined_features = hstack([
            tfidf_features,
            csr_matrix(scaled_aspect_features)
        ])

        return combined_features

    def get_feature_names(self):
        """
        Get names of features for interpretation

        Returns:
        --------
        list
            Feature names
        """
        # TF-IDF feature names
        tfidf_names = self.tfidf_vectorizer.get_feature_names_out()

        # Aspect feature names
        aspect_names = self.aspect_analyzer.get_aspect_names()

        return list(tfidf_names) + aspect_names