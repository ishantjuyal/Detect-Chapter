import pandas as pd
import nltk
import re

df = pd.read_csv(r"C:\Users\ishan\OneDrive\Desktop\Projects\Solve\Dataset\JEE_Data.csv", engine = "python")

def do_preprocess(df):
    for index, row in df.iterrows():
        filter_sentence = ''
        sentence = row['eng']

        sentence = re.sub(r'[^\w\s]', '', sentence)

        words = nltk.word_tokenize(sentence)

        for word in words:
            filter_sentence = filter_sentence  + ' ' + str(word).lower()
        
        df.loc[index, 'preprocessed'] = filter_sentence
    df = df[['eng', "preprocessed", 'chapter']]
    df['index'] = list(df.index)
    return(df)

df = do_preprocess(df)
print(df.head())

df.to_csv("JEE_Data.csv", index = False)