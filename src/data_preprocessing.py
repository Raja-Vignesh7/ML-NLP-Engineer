import pandas as pd
import numpy as np
import string

from datasets import Dataset

from nltk.corpus import stopwords,words
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))  # List of common stopwords
punctuation = string.punctuation  # List of punctuation
valid_words = set(words.words())  


dataset = pd.read_csv('/kaggle/input/steam-reviews/output.csv')
print(dataset.info())

dataset.dropna(subset=['content','is_positive'],inplace=True)

text_labels = list(dataset['is_positive'].unique())
text_labels


class Text_preprocessor:

    def __init__(self,df):
        self.df_copy = df.copy()

    def label_target(self):
        labels = []
        i=0
        for index, row in self.df_copy.iterrows():
            if i%50000==0:
                print(f"convert labels: {i}")
            if row['is_positive'] == 'Positive':
                labels.append(1)
            else:
                labels.append(0)
            i+=1
        self.df_copy['target'] = labels

    
    def preprocess_text(self):
        print('Started labeling target\n')
        self.label_target()
        print('Labeling Completed')
        print('Starting text preprocessing\n')
        i=0
        try:
            processed_content = []

            for index, row in self.df_copy.iterrows():
                if i%50000==0:
                    print(f"no of rows preprocessed: {i}")
                i+=1
                text = row['content']

                if not isinstance(text, str):
                    processed_content.append(None)
                    continue

                text_data = text.lower()

                text_data = ''.join([char for char in text_data if char not in punctuation and char != '·'])

                words = word_tokenize(text_data)

                words = [word for word in words if word not in stop_words or word in valid_words]
                if len(words) >= 4:
                    tokens = ' '.join(words)
                else:
                    tokens = None
                processed_content.append(tokens)

            self.df_copy['tokens'] = processed_content
        except Exception as e:
            print(f'error: {e}')
        print('preprocessing complete\n')
        self.df_copy.dropna(subset=['tokens'],inplace=True)
        return self.df_copy[['id','content','tokens','target']]
    
    
preprocessor = Text_preprocessor(dataset)
processed_dataset = preprocessor.preprocess_text()
processed_dataset

HGF_dataset = Dataset.from_pandas(processed_dataset)
HGF_dataset