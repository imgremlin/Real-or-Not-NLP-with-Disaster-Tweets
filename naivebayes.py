import pandas as pd
import numpy as np
import re
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from wordcloud import STOPWORDS

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', 10)

train = pd.read_csv('train.csv', index_col='id')
test = pd.read_csv('test.csv', index_col='id')
for df in [train, test]:
    
    df.drop(columns=['keyword', 'location'], inplace=True)
    
    text="@Kiwi_Karyn Check out what's in my parking lot!! He said that until last year it was an ambulance in St Johns. http://t.co/hPvOdUD7iP"
    text1="Check out what's in my parking lot He said that until last year it was an ambulance in St Johns"
    
    #number of characters
    df['NumChar'] = df['text'].apply(len)
    
    #to lower
    df['text'] = df['text'].apply(lambda x: x.lower())
    
    #count links
    df['Links'] = df['text'].apply(lambda x: len(re.findall(r"http://", x)))
    
    #del links
    df['text'] = df['text'].apply(lambda x: re.sub(r"https*://\S+","", x))
    
    #count @
    df['Mentions'] = df['text'].apply(lambda x: len(re.findall(r"@", x)))
    
    #count #
    df['Hashtags'] = df['text'].apply(lambda x: len(re.findall(r"#", x)))
    
    #del @#
    df['text'] = df['text'].apply(lambda x: re.sub(r"[@#]","", x))
    
    #count punctuation
    df['Punct'] = df['text'].apply(lambda x: len(re.findall(r"[!#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", x)))
    
    #remove punctuation
    df['text'] = df['text'].apply(lambda x: re.sub(r"[!#$%&()*+,-./:;<=>?@[\]^_{|}~]"," ", x))
    df['text'] = df['text'].apply(lambda x: re.sub(r"' | '|^'|'$", " ", x))
    
    #del lots of spaces
    df['text'] = df['text'].apply(lambda x: re.sub(r"\s{2,}"," ", x))
    
    #word count
    df['WordCount'] = df['text'].apply(lambda x: len(re.split(r" ", x)))
    
    #average word lenght
    df['WordLenght'] = df['text'].apply(lambda x: round(np.mean([len(w) for w in str(x).split()]), 1))
    
    #number of stopwords
    df['StopWordsCount'] = df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    
    text2="Big Top Burning The True Story Of An Arsonist A Missing Girl Â‰Ã›  "
    
    #delete special characters
    df['text'] = df['text'].apply(lambda x: re.sub(r"[^a-z 0-9']", '', x))

    #expand abbreviations
    df['text'] = df['text'].apply(lambda x: re.sub(r"'s", ' is', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r"'ve", ' have', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r"'m", ' am', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r"n't", 'n not', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r"'re", ' are', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r"'d", ' would', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r"'ll", ' will', x))

from sklearn.feature_extraction.text import TfidfVectorizer

#vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer(
                             #preprocessor=clean_text,
                             ngram_range=(1, 1))
#better with no stopwords

#stop_words="english"
#better to expand abbreviations

training_features = vectorizer.fit_transform(train['text'])    
test_features = vectorizer.transform(test['text'])

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=4, random_state=0)

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB


xgb = XGBClassifier(objective ='reg:squarederror')
catb = CatBoostClassifier(verbose=False)
lgbm = LGBMClassifier()
mnd = MultinomialNB(alpha=0.9, fit_prior=True)

#matrix = training_features.todense()


results = cross_val_score(mnd, training_features,
                          train['target'], cv=kfold, scoring='accuracy')
print(f"res mnd: {results.mean():.5f}")
 
'''
mnd.fit(training_features, train['target'])
preds = mnd.predict(test_features)

submission = pd.read_csv('sample_submission.csv')
submission["target"] = preds
'''
#submission.to_csv("submission.csv", index=False)

#train.to_csv('output.csv')
