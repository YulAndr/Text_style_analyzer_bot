import sqlite3
import os
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

conn = sqlite3.connect('text_classificator.db')
cursor = conn.cursor()
cursor.execute("SELECT link FROM Texts")
allwords = []
for row in cursor.fetchall():
    file_path = row[0]
    link = file_path
    current_file = open(link, encoding='utf-8')
    current_content = current_file.read()

    current_content = current_content.lower()
    current_content = re.sub("[0-9]", "", current_content)

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(current_content)

    swf = open('stopwords.txt')
    stopwordsList = swf.read()
    cleared = [word for word in tokens if word not in stopwordsList]

    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in cleared]

    allwords = allwords + stemmed

columns = set(allwords)

import pandas as pd

dictionary = {'text_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
data = list()
for column in columns:
    dictionary[column] = []
    cursor.execute("SELECT link FROM Texts")
    for row in cursor.fetchall():

        stemmed.clear()
        cleared.clear()
        file_path = row[0]
        link = file_path
        current_file = open(link, encoding='utf-8')
        current_content = current_file.read()

        #Отчистка текущего текста
        current_content = current_content.lower()
        current_content = re.sub("[0-9]", "", current_content)

        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(current_content)

        swf = open('stopwords.txt')
        stopwordsList = swf.read()
        cleared = [word for word in tokens if word not in stopwordsList]

        stemmer = PorterStemmer()
        stemmed = [stemmer.stem(word) for word in cleared]

        #Подсчет слов
        dictionary[column].append(stemmed.count(column))

Frequency_Dictionary = pd.DataFrame(dictionary)
Frequency_Dictionary.to_csv('S_Bojei_Pomoshchyu.csv')
