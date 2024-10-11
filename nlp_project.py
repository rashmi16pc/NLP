import pandas as pd
import os
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from contextlib import ExitStack
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import textstat

stop_words = []

file_paths = [r"C:\Users\rashm\Downloads\StopWords-20240717T083314Z-001\StopWords\StopWords_Auditor.txt", r"C:\Users\rashm\Downloads\StopWords-20240717T083314Z-001\StopWords\StopWords_Currencies.txt", r"C:\Users\rashm\Downloads\StopWords-20240717T083314Z-001\StopWords\StopWords_DatesandNumbers.txt", r"C:\Users\rashm\Downloads\StopWords-20240717T083314Z-001\StopWords\StopWords_Generic.txt", r"C:\Users\rashm\Downloads\StopWords-20240717T083314Z-001\StopWords\StopWords_GenericLong.txt", r"C:\Users\rashm\Downloads\StopWords-20240717T083314Z-001\StopWords\StopWords_Geographic.txt", r"C:\Users\rashm\Downloads\StopWords-20240717T083314Z-001\StopWords\StopWords_Names.txt"]

with ExitStack() as stack:
    files = [stack.enter_context(open(path, "r")) for path in file_paths]
    for file in files:
        # Process each file
        for line in file:
#            line = line.split("|")
            line = line.strip()
        # reading each word        
            stop_words.append(line)
stop_words.extend(stopwords.words('english'))
stop_words = [word.lower() for word in stop_words]

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

file_path = os.path.abspath(r"C:\Users\rashm\Downloads\Input.xlsx")

df = pd.read_excel(file_path)

def load_word_list(file_path):
    with open(file_path, 'r') as file:
        return set(file.read().split())

positive_words = load_word_list(r"C:\Users\rashm\Downloads\MasterDictionary-20240717T083312Z-001\MasterDictionary\positive-words.txt")
negative_words = load_word_list(r"C:\Users\rashm\Downloads\MasterDictionary-20240717T083312Z-001\MasterDictionary\negative-words.txt")

def sentiment_analysis(text):
    tokens = nltk.word_tokenize(text)
    positive_score = sum(1 for word in tokens if word in positive_words)
    negative_score = sum(1 for word in tokens if word in negative_words)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(tokens) + 0.000001)
    return positive_score, negative_score, polarity_score, subjectivity_score

def average_words_per_sentence(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    return len(words) / len(sentences)

def complex_word_count(text):
    tokens = nltk.word_tokenize(text)
    complex_words = [word for word in tokens if textstat.syllable_count(word) > 2]
    return len(complex_words)

def word_count(text):
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return len(cleaned_tokens)

def count_syllables(word):
    endings_to_exclude = ["es", "ed"]
    for ending in endings_to_exclude:
        if word.endswith(ending):
            return textstat.syllable_count(word[:-len(ending)])
    return textstat.syllable_count(word)

def syllable_count_per_word(text):
    tokens = nltk.word_tokenize(text)
    syllables_per_word = [count_syllables(word) for word in tokens]
    return sum(syllables_per_word) / len(tokens)

def personal_pronouns_count(text):
    pronouns = ["I", "we", "my", "ours", "us"]
    pattern = re.compile(r'\b(?!US\b)(?:' + '|'.join(pronouns) + r')\b', re.IGNORECASE)
    matches = pattern.findall(text)
    return len(matches)

def average_word_length(text):
    tokens = nltk.word_tokenize(text)
    word_lengths = [len(word) for word in tokens if word.isalpha()]
    return sum(word_lengths) / len(word_lengths)

def readability_analysis(text):
    average_sentence_length = textstat.avg_sentence_length(text)
    percentage_complex_words = (complex_word_count(text)/word_count(text))*100 
    fog_index = textstat.gunning_fog(text)
    return average_sentence_length, percentage_complex_words, fog_index


# Load your Excel file
file_path = os.path.abspath(r"C:\Users\rashm\Downloads\Input.xlsx")
df = pd.read_excel(file_path)

# Add columns for the new analyses
df['POSITIVE SCORE'] = 0
df['NEGATIVE SCORE'] = 0
df['POLARITY SCORE'] = 0
df['SUBJECTIVITY SCORE'] = 0
df['AVG SENTENCE LENGTH'] = 0
df['PERCENTAGE OF COMPLEX WORDS'] = 0
df['FOG INDEX'] = 0
df['AVG NUMBER OF WORDS PER SENTENCE'] = 0
df['COMPLEX WORD COUNT'] = 0
df['WORD COUNT'] = 0
df['SYLLABLE PER WORD'] = 0
df['PERSONAL PRONOUNS'] = 0
df['AVG WORD LENGTH'] = 0

for index, row in df.iterrows():
    url = row['URL']
    url_id = row['URL_ID']
    
    # Fetch the content of the URL
    content = requests.get(url).text
    
    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(content, features="html.parser")
    
    # Remove all script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    
    # Extract the text from the HTML
    text = soup.get_text()
    
    # Break the text into lines and remove leading and trailing spaces
    lines = (line.strip() for line in text.splitlines())
    
    # Break multi-headlines into separate lines
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    
    # Drop blank lines and join the cleaned text into a single string
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    # Print the cleaned text (optional)
    print(text)
    
    # Preprocess the text if needed
    cleaned_text = preprocess_text(text)
    
    # Perform analyses
    pos_score, neg_score, pol_score, subj_score = sentiment_analysis(cleaned_text)
    avg_sentence_len, perc_complex_words, fog_idx = readability_analysis(cleaned_text)
    avg_words_sentence = average_words_per_sentence(cleaned_text)
    complex_words_cnt = complex_word_count(cleaned_text)
    word_cnt = word_count(cleaned_text)
    syllables_per_word = syllable_count_per_word(cleaned_text)
    pronouns_cnt = personal_pronouns_count(cleaned_text)
    avg_word_len = average_word_length(cleaned_text)
    
    # Update the DataFrame
    df.at[index, 'POSITIVE SCORE'] = pos_score
    df.at[index, 'NEGATIVE SCORE'] = neg_score
    df.at[index, 'POLARITY SCORE'] = pol_score
    df.at[index, 'SUBJECTIVITY SCORE'] = subj_score
    df.at[index, 'AVG SENTENCE LENGTH'] = avg_sentence_len
    df.at[index, 'PERCENTAGE OF COMPLEX WORDS'] = perc_complex_words
    df.at[index, 'FOG INDEX'] = fog_idx
    df.at[index, 'AVG NUMBER OF WORDS PER SENTENCE'] = avg_words_sentence
    df.at[index, 'COMPLEX WORD COUNT'] = complex_words_cnt
    df.at[index, 'WORD COUNT'] = word_cnt
    df.at[index, 'SYLLABLE PER WORD'] = syllables_per_word
    df.at[index, 'PERSONAL PRONOUNS'] = pronouns_cnt
    df.at[index, 'AVG WORD LENGTH'] = avg_word_len

# Save the DataFrame with the new columns to a new Excel file
output_file_path = os.path.abspath(r"C:\Users\rashm\Downloads\Output.xlsx")
df.to_excel(output_file_path, index=False)