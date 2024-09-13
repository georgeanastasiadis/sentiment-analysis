import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences



def nltk_data_download():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

def text_preprocessing(text):

        lemmatizer= WordNetLemmatizer()

        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = text.lower()
        
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word, pos='v') for word in words if word not in set(stopwords.words('english'))]
        text = ' '.join(words)

        return text

def data_preparation(text, tokenizer):
    data = pd.Series(text, name='text')

    X_sequence = tokenizer.texts_to_sequences(data)
    X_sequence_padded = pad_sequences(X_sequence, maxlen=79)

    return X_sequence_padded

def preprocess_dataframe(df, column_name):
    
    df[column_name] = df[column_name].apply(text_preprocessing)
    return df