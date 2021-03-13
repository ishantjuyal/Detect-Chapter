from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from ktrain import text
import pandas as pd
import ktrain
import nltk
import re

train = pd.read_csv("https://raw.githubusercontent.com/ishantjuyal/Solve/main/Dataset/train_JEE.csv", engine = "python")


# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')


stop_words = stopwords.words('english')
lemmatizer=WordNetLemmatizer()

def do_preprocess(df):
    for index, row in df.iterrows():
        filter_sentence = ''
        sentence = row['eng']

        # Cleaning the sentence with regex
        sentence = re.sub(r'[^\w\s]', '', sentence)

        # Tokenization
        words = nltk.word_tokenize(sentence)

        # Stopwords removal
        words = [w for w in words if not w in stop_words]
        
        for words in words:
            filter_sentence = filter_sentence  + ' ' + str(lemmatizer.lemmatize(words)).lower()
        
        df.loc[index, 'eng'] = filter_sentence
    df = df[['eng', 'chapter']]
    return(df)

train_preprocessed = do_preprocess(train)


(X_train, y_train), (X_test, y_test), preprocess = text.texts_from_df(train_df= train_preprocessed, 
                                                                      text_column = 'eng',
                                                                      label_columns = 'chapter',
                                                                      maxlen = 256,
                                                                      preprocess_mode = 'bert')

model = text.text_classifier(name= 'bert',
                             train_data= (X_train, y_train),
                             preproc = preprocess)


learner = ktrain.get_learner(model = model,
                             train_data = (X_train, y_train),
                             val_data = (X_test, y_test),
                             batch_size = 18)

learner.fit_onecycle(lr = 2e-5, epochs = 1)

predictor = ktrain.get_predictor(learner.model, preproc= preprocess)

def preprocess_sentence(question):
    filter_sentence = ''
    sentence = question

    # Cleaning the sentence with regex
    sentence = re.sub(r'[^\w\s]', '', sentence)

    # Tokenization
    words = nltk.word_tokenize(sentence)
        
    # Stopwords removal
    words = [w for w in words if not w in stop_words]
        
    for words in words:
        filter_sentence = filter_sentence  + ' ' + str(lemmatizer.lemmatize(words)).lower()
    
    return(filter_sentence)

def detect(question):
    question = preprocess_sentence(question)
    chapter = predictor.predict(question)
    return('The chapter is ' + chapter)