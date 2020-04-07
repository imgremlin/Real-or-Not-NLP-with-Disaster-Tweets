import pandas as pd
import numpy as np
import re
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from wordcloud import STOPWORDS
import operator

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
    df['text'] = df['text'].apply(lambda x: re.sub(r"[^a-z ']", '', x))

    #expand abbreviations
    df['text'] = df['text'].apply(lambda x: re.sub(r"'s", ' is', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r"'ve", ' have', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r"'m", ' am', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r"n't", 'n not', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r"'re", ' are', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r"'d", ' would', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r"'ll", ' will', x))


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['text'])

MAXLEN=100

train_sequences = tokenizer.texts_to_sequences(train['text'])
train_data = pad_sequences(train_sequences, maxlen=MAXLEN)

test_sequences = tokenizer.texts_to_sequences(test['text'])
test_data = pad_sequences(test_sequences, maxlen=100)

word_index=tokenizer.word_index
print('Number of unique words:',len(word_index))

embeddings_index = dict()

GLOVE_DIM=200
f = open('embeddings/glove.twitter.27B.200d.txt', encoding="utf8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

num_words=len(word_index)+1


embedding_matrix = np.zeros((num_words, GLOVE_DIM))

for word, index in tokenizer.word_index.items():
    if index > num_words - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            
            
def build_vocab(X):
    
    tweets = X.apply(lambda s: s.split()).values      
    vocab = {}
    for tweet in tweets:
        for word in tweet:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1                
    return vocab

def check_embeddings_coverage(X, embeddings):
    
    vocab = build_vocab(X)    
    covered = {}
    oov = {}    
    n_covered = 0
    n_oov = 0
    for word in vocab:
        try:
            covered[word] = embeddings[word]
            n_covered += vocab[word]
        except:
            oov[word] = vocab[word]
            n_oov += vocab[word]
    vocab_coverage = len(covered) / len(vocab)
    text_coverage = (n_covered / (n_covered + n_oov))
    sorted_oov = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
    return sorted_oov, vocab_coverage, text_coverage


train_glove_oov, train_glove_vocab_coverage, train_glove_text_coverage = check_embeddings_coverage(train['text'], embeddings_index)
test_glove_oov, test_glove_vocab_coverage,test_glove_text_coverage = check_embeddings_coverage(test['text'], embeddings_index)
print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set'.format(train_glove_vocab_coverage, train_glove_text_coverage))
print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Test Set'.format(test_glove_vocab_coverage, test_glove_text_coverage))

#print(test_glove_oov)

from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, SpatialDropout1D
from keras.layers import Dropout, Bidirectional, Activation
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding

model = Sequential()
model.add(Embedding(num_words, GLOVE_DIM, input_length=100, weights=[embedding_matrix],
                    trainable=False))
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(300))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(train_data, train['target'], validation_split=0.25, epochs=5)
model_loss = pd.DataFrame(model.history.history)
model_loss[['accuracy','val_accuracy']].plot()

predictions= model.predict_classes(test_data)

#print(predictions)

submission = pd.read_csv('sample_submission.csv')
submission["target"] = predictions
submission.to_csv("submission_glove.csv", index=False)
