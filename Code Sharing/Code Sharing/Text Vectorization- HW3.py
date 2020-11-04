import pandas as pd
import numpy as np
import time
import re
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import os
#Read the data
#---------------Restaurant Reviews data -------------------------------------------
path_rest = 'D:/Text Mining/Data Files/Restaraunt/Restaurant_Reviews.tsv'
restr_reviews = pd.read_csv(path_rest, sep = '\t')

restr_reviews.head(5)


#Get all the folders
path_root = 'D:/Text Mining/Data Files/aclImdb_v1/aclImdb/'

label = []
daty = []
folders = []
for p in next(os.walk(path_root))[1]:
    d = next(os.walk(path_root+p))[1]
    for i in range(len(d)):
        label.append(d[i])
        daty.append(p)
        folders.append(path_root+p+'/'+d[i])

#Generate the list of files and the label
flist = []        
for fol in folders:
    flist.append((next(os.walk(fol)))[2])


dictx = {'label': label, 'type': daty, 'folder': folders, 'list': flist}

df=pd.DataFrame(dictx)

df.head(5)

#read test dataset
test_pos_files = df[(df['label'] == 'pos') & (df['type'] == 'test')]['list']
path_name = df[(df['label'] == 'pos') & (df['type'] == 'test')]['folder'].values

pos_test = []
pos_lab = []
for d in next(os.walk(path_name[0])):
    for f in d:
        if f.endswith('.txt') == True:
            with open(path_name[0]+'/'+str(f), 'r', encoding="utf8") as filecon:
                filss= filecon.readlines()
            pos_test.append(filss[0])
            pos_lab.append('pos')

test_neg_files = df[(df['label'] == 'neg') & (df['type'] == 'test')]['list']
path_name = df[(df['label'] == 'neg') & (df['type'] == 'test')]['folder'].values

for d in next(os.walk(path_name[0])):
    for f in d:
        if f.endswith('.txt') == True:
            with open(path_name[0]+'/'+str(f), 'r', encoding="utf8") as filecon:
                filss= filecon.readlines()
            pos_test.append(filss[0])
            pos_lab.append('neg')


dictss = {'label': pos_lab, 'review': pos_test}

test_df = pd.DataFrame(dictss)
    



#Training data set
train_pos_files = df[(df['label'] == 'pos') & (df['type'] == 'train')]['list']
path_name = df[(df['label'] == 'pos') & (df['type'] == 'train')]['folder'].values

pos_train = []
pos_lab = []
for d in next(os.walk(path_name[0])):
    for f in d:
        if f.endswith('.txt') == True:
            with open(path_name[0]+'/'+str(f), 'r', encoding="utf8") as filecon:
                filss= filecon.readlines()
            pos_train.append(filss[0])
            pos_lab.append('pos')



train_neg_files = df[(df['label'] == 'neg') & (df['type'] == 'train')]['list']
path_name = df[(df['label'] == 'neg') & (df['type'] == 'train')]['folder'].values

for d in next(os.walk(path_name[0])):
    for f in d:
        if f.endswith('.txt') == True:
            with open(path_name[0]+'/'+str(f), 'r', encoding="utf8") as filecon:
                filss= filecon.readlines()
            pos_train.append(filss[0])
            pos_lab.append('neg')


dictss = {'label': pos_lab, 'review': pos_train}

train_df = pd.DataFrame(dictss)

#------------------------------ Data Cleansing ------------------------------------------
#Clean the text for punctuations and other special characters
def text_clean(text):
	text = text.replace("\n"," ").replace("\r"," ").replace('&amp', "")
    .replace('br', "").replace('.',"")
	punctuations = '!#$%^&*()~!@<>/;:{}[]"?'
	t = str.maketrans(dict.fromkeys(punctuations, " "))
	text = text.translate(t)
	t = str.maketrans(dict.fromkeys("'`", ""))
	text = text.translate(t)
	return text

#Remove URL
def rem_url(text):
    text = re.sub(r'http\S+', '', text)
    return text

#Frequency distribution

from nltk import FreqDist

def n_grams(textwords, n):
    textdist = FreqDist(textwords)
    textitems = textdist.most_common(n)
    for item in textitems:
        print (item[0], '\t', item[1])
    
    
    #Lets look at the bigrams
    textdist = FreqDist((nltk.bigrams(textwords)))
    textitems = textdist.most_common(n)
    for item in textitems:
        print (item[0], '\t', item[1])
    
    #Lets look at the trigrams
    textdist = FreqDist((nltk.trigrams(textwords)))
    textitems = textdist.most_common(n)
    for item in textitems:
        print (item[0], '\t', item[1])





#------------------------------- Start cleansing the data ------------------------------
        
pos_train = train_df['review']

import nltk
chk_v = s
textwords = []
for i in chk_v:
    i = nltk.word_tokenize(i)
    textwords.extend([word.lower() for word in i])
len(textwords)
n_grams(textwords, 10)

#train_df['review']
#Step 1: Remove URL
d = []
for u in pos_train:
    d.append(rem_url(u))


#Step 2: Remove punctuations and special characters
s = []
for u in d:
    s.append(text_clean(u))


#Lets tokenize

#Regular Expression
def regexp_tkn(text):
    wrd = re.compile(r'\w+')
    words = wrd.findall(text)
    return words

#NTLK tokenizers
import nltk

from nltk.tokenize import TreebankWordTokenizer

from nltk.tokenize import WordPunctTokenizer

from nltk.tokenize import regexp_tokenize 

#------------------------------------------------

#Spacy Tokenizer
import en_core_web_sm
nlp = en_core_web_sm.load(disable = ['ner', 'tagger'])
def spacy_tkn(text):
    tokens = [x.text for x in nlp(text)]
    return tokens

#Keras tokenizer
from keras.preprocessing.text import text_to_word_sequence
def keras_tkn(text):
    tokens = text_to_word_sequence(text)
    return tokens



tkn_time = []
tknizer_name = []
#Tokenize the corpus
t1 = time.time()
tkns_re = []
for u in s:
    tkns_re.append(regexp_tkn(u))    
t2 = time.time()
j = 0
for i in range(0, len(tkns_re)):
    j = j+len(np.unique(np.array(tkns_re[i])))

tkn_time.append('Regular Expressions = %.3f'%(t2-t1)+'    Token Count:'+str(j))
tknizer_name.append('tkns_re')


tkns_nltktb = []
t1 = time.time()
from nltk.tokenize import TreebankWordTokenizer
sd = TreebankWordTokenizer()
for u in s:
    tkns_nltktb.append(sd.tokenize(u))    
t2 = time.time()
j = 0
for i in range(0, len(tkns_nltktb)):
    j = j+len(np.unique(np.array(tkns_nltktb[i])))

tkn_time.append('Treebank Tokenizer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
tknizer_name.append('tkns_nltktb')


tkns_nltkw = []
t1 = time.time()
for u in s:
    tkns_nltkw.append(nltk.word_tokenize(u))    
t2 = time.time()
j = 0
for i in range(0, len(tkns_nltkw)):
    j = j+len(np.unique(np.array(tkns_nltkw[i])))

tkn_time.append('Word Tokenizer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
tknizer_name.append('tkns_nltkw')

tkns_nltkwp = []
t1 = time.time()
for u in s:
    tkns_nltkwp.append(sd.tokenize(u))    
t2 = time.time()
j = 0
for i in range(0, len(tkns_nltkwp)):
    j = j+len(np.unique(np.array(tkns_nltkwp[i])))

tkn_time.append('Word Punct Tokenizer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
tknizer_name.append('tkns_nltkwp')

t1 = time.time()
tkns_spacy = []
for u in s:
    tkns_spacy.append(spacy_tkn(u))    
t2 = time.time()
j = 0
for i in range(0, len(tkns_spacy)):
    j = j+len(np.unique(np.array(tkns_spacy[i])))
tkn_time.append('Spacy Tokenizer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
tknizer_name.append('tkns_spacy')


t1 = time.time()
tkns_keras = []
for u in s:
    tkns_keras.append(keras_tkn(u))
t2 = time.time()
j = 0
for i in range(0, len(tkns_keras)):
    j = j+len(np.unique(np.array(tkns_keras[i])))
tkn_time.append('Keras Tokenizer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
tknizer_name.append('tkns_keras')

for i in tkn_time:
    print(i)
    

#Post tokenization - Stemming and Lemmatization


from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import RegexpStemmer
from nltk.stem.snowball import SnowballStemmer

stemmed_cor_len = []
stemm_lemm_coll = []
t1 = time.time()
ps_Stemmer = PorterStemmer()
stemmed_corpus_re_p = []
for i in range(0, len(tkns_re)):
    k = [ps_Stemmer.stem(w) for w in tkns_re[i]]
    stemmed_corpus_re_p.append(k)
t2 = time.time()
j = 0
for i in range(0, len(stemmed_corpus_re_p)):
    j = j+len(np.unique(np.array(stemmed_corpus_re_p[i])))
stemmed_cor_len.append('Porter Stemmer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
stemm_lemm_coll.append('stemmed_corpus_re_p')

stemmed_corpus_re_s = []
t1 = time.time()
ls_Stemmer = LancasterStemmer()
for i in range(0, len(tkns_re)):
    k = [ls_Stemmer.stem(w) for w in tkns_re[i]]
    stemmed_corpus_re_s.append(k)
t2 = time.time()

j = 0
for i in range(0, len(stemmed_corpus_re_s)):
    j = j+len(np.unique(np.array(stemmed_corpus_re_s[i])))
stemmed_cor_len.append('Lancaster Stemmer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
stemm_lemm_coll.append('stemmed_corpus_re_s')

stemmed_corpus_re_sb = []
t1 = time.time()
sb_Stemmer=SnowballStemmer("english")
for i in range(0, len(tkns_re)):
    k = [sb_Stemmer.stem(w) for w in tkns_re[i]]
    stemmed_corpus_re_sb.append(k)
t2 = time.time()

j = 0
for i in range(0, len(stemmed_corpus_re_sb)):
    j = j+len(np.unique(np.array(stemmed_corpus_re_sb[i])))

stemmed_cor_len.append('Snowball Stemmer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
stemm_lemm_coll.append('stemmed_corpus_re_sb')
#Lemmatizers

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

stemmed_corpus_re_wn = []
t1 = time.time()
for i in range(0, len(tkns_re)):
    k = [lemmatizer.lemmatize(w) for w in tkns_re[i]]
    stemmed_corpus_re_wn.append(k)
t2 = time.time()

j = 0
for i in range(0, len(stemmed_corpus_re_wn)):
    j = j+len(np.unique(np.array(stemmed_corpus_re_wn[i])))
stemmed_cor_len.append('Wordnet Lemmatizer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
stemm_lemm_coll.append('stemmed_corpus_re_wn')

stemmed_corpus_re_pa = []
t1 = time.time()

import pattern.en
from pattern.en import parse
from pattern.en import lemma, lexeme
for i in range(0, len(tkns_re)):
    k = [lemma(w) for w in tkns_re[i]]
    stemmed_corpus_re_pa.append(k)
t2 = time.time()
j = 0
for i in range(0, len(stemmed_corpus_re_pa)):
    j = j+len(np.unique(np.array(stemmed_corpus_re_pa[i])))
stemmed_cor_len.append('Pattern Lemmatizer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
stemm_lemm_coll.append('stemmed_corpus_re_pa')


#Lemmatize and POS Tag

stemmed_corpus_re_ge = []
t1 = time.time()

import gensim
import gensim.utils
from gensim.utils import lemmatize
for i in range(0, len(tkns_re)):
    k = [lemmatize(w) for w in tkns_re[i]]
    stemmed_corpus_re_ge.append(k)
t2 = time.time()


#Split the gensim lemma into tokens and pos tag
docs_tkns = []
docs_pos = []
tokens = []
pos_tag = []
for i in range(0, len(stemmed_corpus_re_ge)):
    f = stemmed_corpus_re_ge[i]
    for j in f:
        if len(j) != 0:
            d = str(j).split('/')
            tokens.append(d[0].replace("[b'", ''))
            pos_tag.append(d[1].replace("']", ''))
    docs_tkns.append(tokens)
    docs_pos.append(pos_tag)
    tokens = []
    pos_tag = []


j = 0
for i in range(0, len(docs_tkns)):
    j = j+len(np.unique(np.array(docs_tkns[i])))
stemmed_cor_len.append('Gensim Lemmatizer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
stemm_lemm_coll.append('docs_tkns')

from textblob import TextBlob, Word
stemmed_corpus_re_tb = []
t1 = time.time()
for i in range(0, len(tkns_re)):
    k = [Word(w).lemmatize() for w in tkns_re[i]]
    stemmed_corpus_re_tb.append(k)
t2 = time.time()
j = 0
for i in range(0, len(stemmed_corpus_re_tb)):
    j = j+len(np.unique(np.array(stemmed_corpus_re_tb[i])))
stemmed_cor_len.append('TextBlob Lemmatizer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
stemm_lemm_coll.append('stemmed_corpus_re_tb')


for i in stemmed_cor_len:
    print(i)


#-----------------Test-------------

d_test = []
for u in test_df['review']:
    d_test.append(rem_url(u))

#Step 2: Remove punctuations and special characters
s_test = []
for u in d_test:
    s_test.append(text_clean(u))


#Tokenize the corpus
t1 = time.time()

tkns_re_test = []
for u in s_test:
    tkns_re_test.append(regexp_tkn(u))    
t2 = time.time()
j = 0
for i in range(0, len(tkns_re_test)):
    j = j+len(np.unique(np.array(tkns_re_test[i])))

tkn_time.append('Regular Expressions Test = %.3f'%(t2-t1)+'    Token Count:'+str(j))
tknizer_name.append('tkns_re_test')


#Lemmatize the corpus

t1 = time.time()
ps_Stemmer = PorterStemmer()
tstemmed_corpus_re_p = []
for i in range(0, len(tkns_re_test)):
    k = [ps_Stemmer.stem(w) for w in tkns_re_test[i]]
    tstemmed_corpus_re_p.append(k)
t2 = time.time()
j = 0
for i in range(0, len(tstemmed_corpus_re_p)):
    j = j+len(np.unique(np.array(tstemmed_corpus_re_p[i])))
stemmed_cor_len.append('Porter Stemmer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
stemm_lemm_coll.append('stemmed_corpus_re_p')

tstemmed_corpus_re_s = []
t1 = time.time()
ls_Stemmer = LancasterStemmer()
for i in range(0, len(tkns_re_test)):
    k = [ls_Stemmer.stem(w) for w in tkns_re_test[i]]
    tstemmed_corpus_re_s.append(k)
t2 = time.time()

j = 0
for i in range(0, len(tstemmed_corpus_re_s)):
    j = j+len(np.unique(np.array(tstemmed_corpus_re_s[i])))
stemmed_cor_len.append('Lancaster Stemmer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
stemm_lemm_coll.append('stemmed_corpus_re_s')

tstemmed_corpus_re_sb = []
t1 = time.time()
sb_Stemmer=SnowballStemmer("english")
for i in range(0, len(tkns_re_test)):
    k = [sb_Stemmer.stem(w) for w in tkns_re_test[i]]
    tstemmed_corpus_re_sb.append(k)
t2 = time.time()

j = 0
for i in range(0, len(tstemmed_corpus_re_sb)):
    j = j+len(np.unique(np.array(tstemmed_corpus_re_sb[i])))

stemmed_cor_len.append('Snowball Stemmer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
stemm_lemm_coll.append('stemmed_corpus_re_sb')
#Lemmatizers

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

tstemmed_corpus_re_wn = []
t1 = time.time()
for i in range(0, len(tkns_re_test)):
    k = [lemmatizer.lemmatize(w) for w in tkns_re_test[i]]
    tstemmed_corpus_re_wn.append(k)
t2 = time.time()

j = 0
for i in range(0, len(tstemmed_corpus_re_wn)):
    j = j+len(np.unique(np.array(tstemmed_corpus_re_wn[i])))
stemmed_cor_len.append('Wordnet Lemmatizer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
stemm_lemm_coll.append('stemmed_corpus_re_wn')

tstemmed_corpus_re_pa = []
t1 = time.time()
import pattern.en
from pattern.en import parse
from pattern.en import lemma, lexeme
for i in range(0, len(tkns_re_test)):
    k = [lemma(w) for w in tkns_re_test[i]]
    tstemmed_corpus_re_pa.append(k)
t2 = time.time()
j = 0
for i in range(0, len(tstemmed_corpus_re_pa)):
    j = j+len(np.unique(np.array(tstemmed_corpus_re_pa[i])))
stemmed_cor_len.append('Pattern Lemmatizer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
stemm_lemm_coll.append('stemmed_corpus_re_pa')


#Lemmatize and POS Tag
tstemmed_corpus_re_ge = []
t1 = time.time()
import gensim
import gensim.utils
from gensim.utils import lemmatize
for i in range(0, len(tkns_re_test)):
    k = [lemmatize(w) for w in tkns_re_test[i]]
    tstemmed_corpus_re_ge.append(k)
t2 = time.time()


#Split the gensim lemma into tokens and pos tag
tdocs_tkns = []
tdocs_pos = []
tokens = []
pos_tag = []
for i in range(0, len(tstemmed_corpus_re_ge)):
    f = tstemmed_corpus_re_ge[i]
    for j in f:
        if len(j) != 0:
            d = str(j).split('/')
            tokens.append(d[0].replace("[b'", ''))
            pos_tag.append(d[1].replace("']", ''))
    tdocs_tkns.append(tokens)
    tdocs_pos.append(pos_tag)
    tokens = []
    pos_tag = []


j = 0
for i in range(0, len(tdocs_tkns)):
    j = j+len(np.unique(np.array(tdocs_tkns[i])))
stemmed_cor_len.append('Gensim Lemmatizer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
stemm_lemm_coll.append('docs_tkns')

from textblob import TextBlob, Word
tstemmed_corpus_re_tb = []
t1 = time.time()
for i in range(0, len(tkns_re_test)):
    k = [Word(w).lemmatize() for w in tkns_re_test[i]]
    tstemmed_corpus_re_tb.append(k)
t2 = time.time()
j = 0
for i in range(0, len(tstemmed_corpus_re_tb)):
    j = j+len(np.unique(np.array(tstemmed_corpus_re_tb[i])))
stemmed_cor_len.append('TextBlob Lemmatizer = %.3f'%(t2-t1)+'    Token Count:'+str(j))
stemm_lemm_coll.append('stemmed_corpus_re_tb')


for i in stemmed_cor_len:
    print(i)

#Lets check the frequency distribution


import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import pandas as pd
#Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
    
chk_v = docs_tkns
textwords = []
for i in chk_v:
    textwords.extend([word.lower() for word in i])
n_grams(textwords, 10)

#Lemma to tfidf vectorizer

#restr_reviews.columns

train_df.columns
y_train = train_df['label']

corpus = []
sep = ' '
len2 = []
len3 = []
len4 =[]
len5 = []

for i in docs_tkns:
    s = [w for w in i if len(w)>2]
    for w in i:
        if len(w) == 2:
            len2.append(w)
        elif len(w) == 3:
            len3.append(w)
        elif len(w) == 4:
            len4.append(w)
        else:
            len5.append(w)
    corpus.append(sep.join(s))

len(len2)+len(len3)+len(len4)+len(len5)
print('Lenth 2 Tokens: ' + str(len(len2)))
print('Lenth 3 Tokens: ' + str(len(len3)))
print('Lenth 4 Tokens: ' + str(len(len4)))
print('Lenth 5 or more Tokens: ' + str(len(len5)))

sd =pd.DataFrame(len5, columns = ['Name'])
sd.columns
sd['Name'].value_counts()

cv= TfidfVectorizer(binary = False, use_idf = True, smooth_idf = False, lowercase = True, 
                    stop_words = stop_words, min_df=1, max_df = 0.75, ngram_range=(1,3))

#dfs = pd.DataFrame(cv.fit_transform(corpus).toarray(), columns = cv.get_feature_names())

cv = cv.fit(corpus)

x_train = cv.transform(corpus)

y_test = test_df['label']

corpus = []
len2 = []
len3 = []
len4 =[]
len5 = []
sep = ' '
for i in tdocs_tkns:
    s = [w for w in i if len(w)>2]
    for w in i:
        if len(w) == 2:
            len2.append(w)
        elif len(w) == 3:
            len3.append(w)
        elif len(w) == 4:
            len4.append(w)
        else:
            len5.append(w)
    corpus.append(sep.join(s))


x_test = cv.transform(corpus)


#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

#Now get x_test

from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from matplotlib import colors
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


acc = []
alg = []
list_results = []
def run_model(model, alg_name, plot_index):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    list_results.append(y_pred)
    accuracy =  accuracy_score(y_test, y_pred) * 100
    print(alg_name+'              '+str(accuracy))

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
run_model(model, "Multinomial Naive Bayes", 7)


from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
run_model(model, " MLP Neural network ", 8)

from sklearn import tree
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
run_model(model, "Decision Tree", 1)

# ------ SVM Classifier ----------------
from sklearn.svm import SVC
model = SVC()
run_model(model, "SVM Classifier", 4)

# -------- Nearest Neighbors ----------
from sklearn import neighbors
model = neighbors.KNeighborsClassifier()
run_model(model, "Nearest Neighbors Classifier", 5)

# ---------- SGD Classifier -----------------
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
model = OneVsRestClassifier(SGDClassifier())
run_model(model, "SGD Classifier", 6)


import numpy
number_list = list_results[4]
(unique, counts) = numpy.unique(number_list, return_counts=True)
frequencies = numpy.asarray((unique, counts)).T
print(frequencies)
#----------------------------------------------------------------------------