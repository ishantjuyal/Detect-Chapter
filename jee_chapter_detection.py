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

detect("If  f(x)  is a differentiable function and  g(x) is a double differentiable function such that  |f(x)|≤1  and  f'(x)=g(x) . If  f2(0)+g2(0)=9 . Prove that there exists some  c∈(–3,3) such that g(c).g''(c)<0.")

detect("The pH of a 0.1 M monobasic acid solution is found to be 2. The osmotic pressure of this solution at a temperature of T(K) is")

detect("If the radius of the earth were to shrink by one percent, its mass remaining the same, the acceleration due to gravity on the earth’s surface would")

detect("A wall has two layers A and B, each made of different material. Both the layers have the same thickness. The thermal conductivity of the material of A is twice that of B. Under thermal equilibrium, the temperature difference across the wall is 36° C  The temperature difference across the layer A is")

detect("Two bodies M and N of equal masses are suspended from two separate massless springs of spring con­stants k1 and k2 respectively. If the two bodies os­cillate vertically such that their maximum veloci­ties are equal, the ratio of the amplitude of vibra­tion of M to that of N is")

detect("Two equal point charges are fixed at x = -a and x = +a on the axis. Another point charge Q is placed at the origin. The change in the electrical potential energy of Q, when it is displaced by a small distance x along the x-axis, is approximately proportional to")

detect("A particle of mass m is moving in a circular path of constant radius r such that its centripetal acceleration ac is varying with time ‘t’ as ac = k2rt2 where ‘k’ is a constant. The power delivered to the particle by the force acting on it is")

detect("A smooth sphere A is moving on a frictionless horizontal plane with angular speed co and centre of mass velocity v. It collides elastically and head on with an identical sphere B at rest. Neglect friction everywhere. After the collision, their angular speeds are (0A and 0)B, respectively. Then")

detect("if the chord y = mx + 1 of the circle x2 + y2 = 1 subtend an angle of 45 degree as the major segment of the circle, then the value of m is")

