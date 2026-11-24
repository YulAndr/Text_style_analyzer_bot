# English Text Style Classification — NLP-based Telegram Bot

## Project Overview

This project was developed by a team of four as an exploration of automatic functional style classification in English using natural language processing (NLP) and basic machine learning techniques.

The aim was to investigate how differences in lexical distribution across functional styles could be used to automatically classify written texts. The resulting application was implemented as a Telegram bot that allowed users to submit English texts and receive a predicted style category.

The project focuses on four functional styles:

- Literary writing (fiction)
- Journalistic writing (news)
- Scientific writing
- Official and administrative writing

## Data Collection

A collection of English-language texts was compiled from different sources to represent the four functional styles. The materials included scientific articles, literary works such as books and fairy tales, and other texts selected according to their functional style.

The collected texts were organized in an SQLite database for subsequent linguistic analysis and preprocessing.

## Text Preprocessing and Lexical Analysis

The text preprocessing pipeline was implemented in Python using NLTK, pandas, and regular expressions.

The preprocessing steps included:

- Conversion to lowercase
- Removal of digits and punctuation
- Regular expression-based tokenization
- Stop-word removal
- Stemming using the Porter stemming algorithm
- Word frequency calculation

A frequency dictionary was constructed to examine the distribution of lexical items across texts and identify vocabulary characteristic of different functional styles. This resource was intended to support feature selection for machine learning-based classification.

## Classification Approach

The classification component explores the use of lexical features to distinguish between functional styles. It uses CountVectorizer for feature representation and Logistic Regression for classification, implemented with scikit-learn.

The underlying idea is that different functional styles exhibit different patterns of word usage, which can provide clues about the style of an unseen text.

The project was developed as an experimental prototype rather than a production-ready classification system.

## My Contribution

Within the team, I was responsible for linguistic data collection, organization, preprocessing, and lexical frequency analysis.

My individual contributions included:

- Collecting English-language texts representing different functional styles.
- Creating an SQLite database to store and organize the collected linguistic data.
- Implementing text preprocessing in Python using NLTK and regular expressions.
- Calculating word frequencies and constructing a frequency dictionary using pandas.
- Preparing linguistic data and lexical features for subsequent machine learning tasks.

My work focused on the data preparation and linguistic analysis stages of the project. Model development and training were carried out by another team member.

## Technologies and Tools

| Technology | Application |
|---|---|
| Python | Text processing and application development |
| SQLite | Linguistic data storage and organization |
| NLTK | Natural language processing and stemming |
| Regular expressions | Text cleaning and tokenization |
| pandas | Data processing and frequency dictionary construction |
| scikit-learn | Feature vectorization and text classification |
| Telegram | User interface |

## Repository Structure

| File / Directory | Description |
|---|---|
| `texts/` | Collection of source texts |
| `text_classificator.db` | SQLite database containing linguistic data |
| `preprocessing.py` | Text preprocessing and word frequency extraction |
| `stopwords.txt` | Stop-word list |
| `Frequency_dictionary.csv` | Frequency dictionary in CSV format |
| `frequency_dictionary.xlsx` | Frequency dictionary in spreadsheet format |
| `styleAnalizerCode.py` | Classification component and console-based interface |
