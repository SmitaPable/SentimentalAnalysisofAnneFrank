# Sentiment Analysis of Historical Data: Anne Frank's Diary

This project focuses on performing sentiment analysis on Anne Frank's diary, a historical dataset. The aim is to gain insights into the emotions expressed throughout the diary and analyze the sentiment of the text.

## Dataset
The dataset used in this project is Anne Frank's diary, which provides a unique perspective on historical events and personal experiences during the Holocaust. The text data was extracted from the diary, cleaned to remove artifacts, and prepared for analysis.

## Data Cleaning
The following steps were undertaken during data cleaning:
- **Text Extraction**: The text was extracted from the diary, possibly digitized using Optical Character Recognition (OCR).
- **Cleaning**: The text data was cleaned to remove OCR artifacts, non-textual content, and inconsistencies.
- **Tokenization**: The text was broken down into sentences or words (tokens) for analysis.
- **Normalization**: Text was converted to lowercase, punctuation was removed, and spelling errors were corrected.
- **Stop Word Removal**: Common words that do not contribute to sentiment were removed.


## Emotion Extraction

Emotions are extracted using the NRC Emotion Lexicon, an external lexicon containing words annotated with their associated emotions. Emotions are identified based on the presence of these words in the text.

## Sentiment Analysis

Sentiment analysis involves the following steps:

1. **Calculating Sentiment Score**: The sentiment score is determined by summing the occurrences of positive and negative emotions in the text.
2. **Labeling Sentiment**: Sentiment labels such as "Highly Positive","Positive", "Negative", "Highly Negative" or "Neutral" are assigned based on the sentiment score.

## Date Pair Analysis

Date pair analysis includes:

1. **Identifying Date Pairs**: Date pairs are identified within the diary text.
2. **Extracting Emotions and Sentiment Scores**: Emotions and sentiment scores are extracted for the text between each pair of dates.

## Output

The extracted emotions data, sentiment scores, and labels are written to a CSV file for further analysis.


## Visualization
The cleaned and labeled dataset was visualized using Python's Streamlit library. Interactive graphs and charts were created to explore the emotional journey depicted in Anne Frank's diary. The visualization provides a deeper understanding of the sentiments expressed throughout the diary and enriches the exploration of its historical significance.Here are some graph charts and wordcloud
Streamlit App: https://sentimentalanalysisofannefrank.streamlit.app/
## **Graph 1: Emotion Counts over Time**
![image](https://github.com/SmitaPable/SentimentalAnalysisofAnneFrank/assets/146085760/6ad624a2-426d-4a74-9235-72b646e1462a)
**Graph 2: Word Clouds for Emotions**
![image](https://github.com/SmitaPable/SentimentalAnalysisofAnneFrank/assets/146085760/ba2a6592-1e8d-47ae-9573-b26967f05602)
**Graph 3: Animated Time Series**
![image](https://github.com/SmitaPable/SentimentalAnalysisofAnneFrank/assets/146085760/81a83373-4526-4540-b41d-15ca8b6b3ccb)



## Technologies Used
- Python
- Streamlit
- Pandas
- NLTK (Natural Language Toolkit)
- Lexicon Wordlists


## Acknowledgements
- Anne Frank House for providing access to the diary text.
- NLTK and Streamlit open-source communities for their valuable tools and resources.
