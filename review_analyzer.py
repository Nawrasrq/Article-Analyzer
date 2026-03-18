from bs4 import BeautifulSoup
from collections import Counter
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import urllib.parse
import requests
import logging
import time
import nltk
import os
import re


# MARK: ArticleAnalyzer Class
class ArticleAnalyzer:
    """NLP tool for scraping and analyzing article text."""

    _instance_count = 0

    # MARK: Initialization
    def __init__(self, log_file_path: str = "logs/article_analyzer.log") -> None:
        """
        Initialize the ArticleAnalyzer.

        Parameters
        ----------
        log_file_path : str, default "logs/article_analyzer.log"
            Path to the log file, relative to the working directory.
        """
        ArticleAnalyzer._instance_count += 1
        self.instance_id = ArticleAnalyzer._instance_count

        self._log_file_path = log_file_path
        self.aspects: set[str] = set()
        self.lemmatizer = WordNetLemmatizer()
        self.sid = SentimentIntensityAnalyzer()

        self.configure_logging()

        self.logger.info(f"Initialized ArticleAnalyzer instance_{self.instance_id}")

    def configure_logging(self) -> None:
        """Configure instance-specific file logging."""
        os.makedirs(os.path.dirname(self._log_file_path), exist_ok=True)

        logger_name = f"article_analyzer.instance_{self.instance_id}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.FileHandler(self._log_file_path, encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def dispose(self) -> None:
        """Release logging file handlers and clean up instance resources."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    # MARK: Setup
    def _download_nltk_resources(self) -> None:
        """Check and download necessary NLTK resources."""
        resources = [
            "corpora/stopwords",
            "sentiment/vader_lexicon.zip",
            "corpora/wordnet",
        ]

        for resource in resources:
            try:
                nltk.data.find(resource)
                print(f"{resource} already downloaded.")
            except LookupError:
                print(f"{resource} not found. Downloading...")
                try:
                    nltk.download(resource.split("/")[-1])
                except Exception as e:
                    print(f"Error downloading {resource}: {e}")
                    self.logger.error(f"Failed to download NLTK resource '{resource}': {e}")

    # MARK: Core Analysis
    def get_article(self, url: str) -> str:
        """
        Scrape paragraph text from an article URL.

        Parameters
        ----------
        url : str
            The URL of the article to scrape.

        Returns
        -------
        str
            Concatenated paragraph text, or an empty string on failure.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/110.0.0.0 Safari/537.36"
            )
        }
        try:
            self.logger.info(f"Fetching article from: {url}")
            with requests.get(url, headers=headers) as response:
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
                article_text = "\n".join(p.get_text() for p in soup.find_all("p"))

                if not article_text.strip():
                    print("No article text found at the provided URL.")
                    self.logger.warning(f"No paragraph text found at: {url}")
                    return ""

                self.logger.info(f"Fetched {len(article_text)} characters from article")
                return article_text

        except requests.exceptions.RequestException as e:
            print(f"Error fetching the article: {e}")
            self.logger.error(f"HTTP request failed for '{url}': {e}")
            return ""

    def analyze_text(self, text: str, url: str) -> list[str]:
        """
        Analyze article text using word frequency and sentiment techniques.

        Parameters
        ----------
        text : str
            The raw article text to analyze.
        url : str
            The source URL, used to extract and filter the article title words.

        Returns
        -------
        list[str]
            List of sentences split from the article text.
        """
        try:
            # Text preprocessing: lowercase, split, remove stopwords and title words
            words = text.lower().split()
            stop_words = set(stopwords.words("english"))
            tokens = [word for word in words if word.isalpha() and word not in stop_words]

            game_title = (
                url.split("/")[-1].replace("-", " ").replace("article", "").title().strip().lower()
            )
            title_words = set(game_title.split())
            tokens = [word for word in tokens if word not in title_words]

            # Word counts
            total_words = len(tokens)
            unique_words = len(set(tokens))
            print(f"\nTotal Words (excluding stopwords): {total_words}")
            print(f"Unique Words: {unique_words}\n")
            self.logger.info(f"Word counts — total: {total_words}, unique: {unique_words}")

            # Word frequency analysis
            word_count = Counter(tokens)
            most_common_words = word_count.most_common(10)
            print("Top 10 Most Common Words:", most_common_words)

            for word, _ in most_common_words[:3]:
                self.aspects.add(word)

            # Phrase analysis
            self.phrase_analysis(tokens)

            # Overall sentiment
            sentiment_scores = self.sid.polarity_scores(text)
            print("Overall Sentiment Analysis:", sentiment_scores)
            self.logger.info(f"Overall sentiment: {sentiment_scores}")

            sentences = re.split(r"(?<=[.!?]) +", text)
            self.sentiment_based_analysis(sentences, self.aspects)

            return sentences

        except Exception as e:
            self.logger.error(f"Text analysis failed: {e}")
            raise

    def phrase_analysis(self, tokens: list[str]) -> None:
        """
        Extract top bigrams and trigrams and add them to the aspects set.

        Parameters
        ----------
        tokens : list[str]
            Preprocessed, filtered word tokens from the article.
        """
        try:
            for n in (2, 3):
                n_grams = ngrams(tokens, n)
                ngram_counts = Counter(n_grams)
                top_ngrams = ngram_counts.most_common(10)

                if n == 2:
                    for i in range(min(5, len(top_ngrams))):
                        bigram_str = " ".join(top_ngrams[i][0])
                        self.aspects.add(bigram_str)
                elif n == 3:
                    if top_ngrams:
                        trigram_str = " ".join(top_ngrams[0][0])
                        self.aspects.add(trigram_str)

                print(f"\nTop 10 {n}-grams (excluding title):")
                for ngram, freq in top_ngrams:
                    ngram_str = " ".join(ngram)
                    print(f" {ngram_str} - {freq}")
            print()

        except Exception as e:
            self.logger.error(f"Phrase analysis failed: {e}")
            raise

    def sentiment_based_analysis(self, sentences: list[str], aspects: set[str]) -> None:
        """
        Perform aspect-based sentiment analysis for a list of aspects.

        Parameters
        ----------
        sentences : list[str]
            Sentences from the article text.
        aspects : set[str]
            Words or phrases to analyze sentiment for.
        """
        try:
            aspect_sentiments: dict[str, float | None] = {}

            for aspect in aspects:
                aspect_scores = []
                lemmatized_aspect = self.lemmatizer.lemmatize(aspect)

                for sentence in sentences:
                    lemmatized_sentence = " ".join(
                        [self.lemmatizer.lemmatize(word) for word in sentence.lower().split()]
                    )
                    if lemmatized_aspect in lemmatized_sentence:
                        sentiment = self.sid.polarity_scores(sentence)
                        aspect_scores.append(sentiment["compound"])

                if aspect_scores:
                    aspect_sentiments[aspect] = sum(aspect_scores) / len(aspect_scores)
                else:
                    aspect_sentiments[aspect] = None

            print(f"Aspect-Based Sentiment Analysis: {aspects}")
            for aspect, sentiment in aspect_sentiments.items():
                if sentiment is not None:
                    print(f" Avg Sentiment for '{aspect}': {sentiment:.2f}")
                else:
                    print(f" No sentences found for aspect '{aspect}'.")

            self.logger.info(
                f"Aspect-based analysis complete for {len(aspects)} aspects"
            )

        except Exception as e:
            self.logger.error(f"Aspect-based sentiment analysis failed: {e}")
            raise

    # MARK: Orchestration
    def run_analysis(self, url: str) -> None:
        """
        Orchestrate the full article analysis workflow.

        Parameters
        ----------
        url : str
            The URL of the article to fetch and analyze.
        """
        try:
            self.aspects = set()
            start_time = time.time()

            article = self.get_article(url)
            sentences = self.analyze_text(article, url)
            print()

            elapsed_time = time.time() - start_time
            print(f"Total execution time: {elapsed_time:.2f} seconds\n")
            self.logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")

            while True:
                new_aspect_input = input(
                    "Enter new aspect(s) to analyze (comma-separated or 'exit' to quit): "
                )

                if new_aspect_input.lower() == "exit":
                    print("Exiting...")
                    break

                input_aspects = set(
                    [aspect.strip() for aspect in new_aspect_input.split(",") if aspect.strip()]
                )
                self.sentiment_based_analysis(sentences, input_aspects)
                print()

        except Exception as e:
            self.logger.error(f"Analysis run failed: {e}")
            raise


# MARK: Entry Point
def main() -> None:
    """Run the Article Analyzer CLI."""
    analyzer = ArticleAnalyzer()
    try:
        analyzer._download_nltk_resources()

        url = input("\nPaste a url to analyze its text (or 'exit' to quit):\n")

        if url.lower() == "exit":
            print("Exiting...")
            return

        if not urllib.parse.urlparse(url).scheme:
            print("Invalid URL. Please include the protocol (http/https).")
            return

        analyzer.run_analysis(url)

    finally:
        analyzer.dispose()


if __name__ == "__main__":
    main()
