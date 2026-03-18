# Article Analyzer

A Python tool for scraping and analyzing articles from any URL. Uses NLP techniques to extract word frequency, phrase patterns, and sentiment scores — including interactive aspect-based sentiment analysis.

## Features

- Scrapes article text from a provided URL
- Extracts total and unique word counts along with the top 10 most common words
- Identifies top bigrams and trigrams (multi-word phrases)
- Performs overall sentiment analysis using VADER
- Performs aspect-based sentiment analysis on auto-discovered keywords and phrases
- Interactive CLI for exploring additional words or phrases after the initial analysis
- Writes diagnostic logs to `logs/article_analyzer.log`

## Project Structure

```
article-analyzer/
├── review_analyzer.py    # Main script (ArticleAnalyzer class + CLI entry point)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── LICENSE               # MIT License
```

## Prerequisites

- Python 3.8 or higher
- pip

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/article-analyzer.git
cd article-analyzer
```

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux / macOS
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python review_analyzer.py
```

You will be prompted to paste a URL. After the analysis runs, you can enter additional words or phrases for aspect-based sentiment analysis:

```
Enter new aspect(s) to analyze (comma-separated or 'exit' to quit):
difficulty, storm, shoes
```

Type `exit` to quit.

## Example Usage

```
Paste a url to analyze its text (or 'exit' to quit):
https://www.orlandosentinel.com/2024/11/03/hurricane-center-tracks-growing-caribbean-system-that-could-threaten-florida-next-week/

Total Words (excluding stopwords): 753
Unique Words: 224

Top 10 Most Common Words: [('storm', 21), ('next', 17), ('system', 13), ('caribbean', 13), ('tropical', 13), ('atlantic', 12), ('forecast', 11), ('low', 10), ('center', 9), ('couple', 9)]

Top 10 2-grams (excluding title):
 next couple - 7
 tropical storm - 7
 subtropical storm - 6
 low pressure - 6
 western caribbean - 6
 nhc gives - 6
 named storm - 6
 eastern atlantic - 4
 disorganized showers - 4
 development system - 4

Top 10 3-grams (excluding title):
 next couple days - 4
 nhc gives chance - 4
 gives chance development - 4
 chance development next - 4
 development next two - 4
 next two seven - 4
 atlantic named storm - 4
 tropical storm watches - 3
 storm watches warnings - 3
 watches warnings could - 3

Overall Sentiment Analysis: {'neg': 0.063, 'neu': 0.9, 'pos': 0.037, 'compound': -0.9926}
Aspect-Based Sentiment Analysis: {'storm', 'tropical storm', 'next', 'system', 'next couple days', 'low pressure', 'subtropical storm', 'western caribbean', 'next couple'}
 Avg Sentiment for 'storm': -0.42
 Avg Sentiment for 'tropical storm': -0.45
 Avg Sentiment for 'next': -0.41
 Avg Sentiment for 'system': -0.37
 No sentences found for aspect 'next couple days'.
 Avg Sentiment for 'low pressure': -0.63
 Avg Sentiment for 'subtropical storm': -0.59
 Avg Sentiment for 'western caribbean': -0.29
 Avg Sentiment for 'next couple': -0.65

Total execution time: 2.17 seconds

Enter new aspect(s) to analyze (comma-separated or 'exit' to quit): hurricane
Aspect-Based Sentiment Analysis: {'hurricane'}
 Avg Sentiment for 'hurricane': -0.11

Enter new aspect(s) to analyze (comma-separated or 'exit' to quit): exit
Exiting...
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
