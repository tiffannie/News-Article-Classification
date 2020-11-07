#Data Acquisition
import pandas as pd
import numpy as np
import twitter

import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer

#--------------------------------------------------------------------------#
#-----------------------News Articles--------------------------------------#
#--------------------------------------------------------------------------#
#------------------ Code to read today's file into a DF is after this section ---------
import newspaper
from newspaper import Article
import pandas as pd
url = ('https://www.cnn.com/us', 'https://www.msn.com/en-us/news', 'https://news.yahoo.com/us/', 'https://www.bbc.com/news', 'https://foxnews.com', 'https://msnbc.com', 'https://www.theguardian.com/us')

headlines = []
titles = []
authors = []

for u in url:
    print(u)
    cnn_paper = newspaper.build(u) 

    list_of_articles = []
    
    for artiicle in cnn_paper.articles:
        list_of_articles.append(artiicle.url)
    
    len(list_of_articles)
    
    if len(list_of_articles) > 100:
        le = 100
    else:
        le = len(list_of_articles)
    
    
    
    for i in range(0,le):
        content = Article(list_of_articles[i]) 
        content.download()
        try:
            content.parse()  
            titles.append(str(content.title))
            authors.append(str(content.authors)) 
            headlines.append(str(content.text))
        except:
            pass
    

ContentDF = pd.DataFrame({'Title': titles, 'Author': authors, 'Headlines': headlines})
ContentDF.to_csv('News_new6.csv', index = 0)

ContentDF.drop(columns = ['Author'], inplace= True)

#--------------------------- Run this code to create the dataframe from a previously
#generated CSV file ----------------------------------------------------------------

path = 'Add the path to the news file'
ContentDF = pd.read_csv(path+'/News_new6.csv')
ContentDF.drop(columns = ['Author'], inplace= True)

#Add lenth of the document to the dataframe

len_headlines = []

for a in ContentDF['Headlines']:
    len_headlines.append(len(a))

ContentDF['Length'] = len_headlines


import matplotlib.pyplot as plt

#Review histogram to see the distribution of news documents
plt.hist(ContentDF['Length'])

#Data pre-processing

# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
def text_clean(text):
	text = text.replace("\n"," ").replace("\r"," ").replace('&amp', "").replace('br', "").replace('.',"")
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


def word_counts(final_headlines):        
    #find total word count
    wordcount = {}
    
    for i in final_headlines:
        i = i.lower()
        for w in nltk.word_tokenize(i):
            if w not in wordcount.keys():
                wordcount[w] = 1
            else:
                wordcount[w] +=1
                
    #find word count by document.
    doccount ={}
    for ww in wordcount.keys():
        doccount[ww] = 0
    for i in final_headlines:
        i = i.lower()
        f = nltk.word_tokenize(i)
        for wordss in wordcount.keys():
            if wordss in f:
                doccount[wordss] += 1
    return wordcount, doccount




from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    
def word_cloud(text, i, stop_words):
    wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stop_words,
                          max_words=100,
                          max_font_size=50, 
                          random_state=42
                         ).generate(str(text))
    print(wordcloud)
    fig = plt.figure(1)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title('Wordcloud of key')
    plt.savefig('Tweet_Group_'+str(i)+'.png')
    plt.close()
    plt.show()



#Cleaning the data set and prepare the corpus
#Prepare the corpus
pos_train = ContentDF['Headlines']
print('There are total ' + str(len(pos_train))+ ' documents in the corpus')

import nltk

chk_v = pos_train
textwords = []
for i in chk_v:
    i = nltk.word_tokenize(i)
    textwords.extend([word.lower() for word in i])



print('There are total ' + str(len(textwords))+ ' tokens in the corpus')

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

#Check after cleansing
    
chk_v = s
textwords = []
for i in chk_v:
    i = nltk.word_tokenize(i)
    textwords.extend([word.lower() for word in i])

print('There are total ' + str(len(textwords))+ ' tokens in the corpus after cleaning')


n_grams(textwords, 10)

#Tokenize the text
#Regular Expression
def regexp_tkn(text):
    wrd = re.compile(r'\w+')
    words = wrd.findall(text)
    return words

tkns_re = []
for u in s:
    tkns_re.append(regexp_tkn(u))    

#Use PorterStemmer    
from nltk.stem import PorterStemmer

ps_Stemmer = PorterStemmer()
stemmed_corpus_re_p = []
for i in range(0, len(tkns_re)):
    k = [ps_Stemmer.stem(w) for w in tkns_re[i]]
    stemmed_corpus_re_p.append(k)

#Use GENSIM Lemmatizer    
stemmed_corpus_re_ge = []
import gensim
import gensim.utils
from gensim.utils import lemmatize

for i in range(0, len(tkns_re)):
    k = [lemmatize(w) for w in tkns_re[i]]
    stemmed_corpus_re_ge.append(k)


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


#Detokenize and count by token size
    
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


#Append clean text to the dataframe
ContentDF['Clean Text'] = corpus

            
            
#remove words that appear in less than 3 documents.

wordcount, doccount = word_counts(corpus)

iter1dict = {}

for key, value in doccount.items():
    if value > 5:
        iter1dict[key] = value

#remove terms that are smaller than 3 characters
iter2dict = {}

for key, value in iter1dict.items():
    if len(key) >3:
        iter2dict[key] = value



        
cleanheadline = ""
coll = iter2dict.keys()
f = []


for a in corpus:
    rtoken = nltk.word_tokenize(a)
    for w in rtoken:
        if w in iter2dict.keys():
            cleanheadline = cleanheadline + " " + w
    f.append(cleanheadline)
    cleanheadline = ""



newdf1 = pd.DataFrame(f, columns = ['Headlines'])

#Add the tokenized, cleaned and lemmatized text to the dataframe
ContentDF['Lemmatized Text'] = f
#----------------------------------------------------------------------------#
#Verify the cleansed dataset

corpus = []
for i in range(0, len(ContentDF['Lemmatized Text'])):    
    text = ContentDF['Lemmatized Text'][i]
    ##Convert to list from string
    text = text.split()
    text = " ".join(text)
    corpus.append(text)


textwords = []

for i in corpus:
    textwords.extend([word.lower() for sent in nltk.sent_tokenize(i) for word in nltk.word_tokenize(sent)])

tok = pd.DataFrame({'words': textwords})
print ('there are ' + str(tok.shape[0]) + ' items in Corpus')

#unigrams
textdist = FreqDist(textwords)
textitems = textdist.most_common(10)
for item in textitems:
    print (item[0], '\t', item[1])


#Lets look at the bigrams
textdist = FreqDist((nltk.bigrams(textwords)))
textitems = textdist.most_common(10)
for item in textitems:
    print (item[0], '\t', item[1])

#Lets look at the trigrams
textdist = FreqDist((nltk.trigrams(textwords)))
textitems = textdist.most_common(10)
for item in textitems:
    print (item[0], '\t', item[1])

#-------------------------------------------------------------------------#
#Feature extraction
#First Iteration: Lets do Unigram
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
 
X = ContentDF['Lemmatized Text']

#Using tfidf vectorizer
from sklearn.metrics.pairwise import cosine_similarity


tfidf_vectorizer = TfidfVectorizer(max_df=0.90,
                                min_df=0.1,
                                 use_idf=True, ngram_range=(1,3))


tfidf_matrix = tfidf_vectorizer.fit_transform(X) 
#fit the vectorizer to synopses
terms = tfidf_vectorizer.get_feature_names()

from sklearn.cluster import KMeans
num_clusters = 7
km = KMeans(n_clusters=num_clusters)
%time km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

#Save your model to pickle file
import joblib
#from sklearn.externals import joblib

joblib.dump(km,  'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

#Topic Modelling and word clouds

artic = { 'cluster': clusters, 'clean content': X }
frame = pd.DataFrame(artic,  columns = ['cluster', 'clean content'])
frame['cluster'].unique()


#Topic Modelling for Clusters

#Use LDA - Latent Dirichelet Allocation

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
from gensim import corpora, models

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        result.append(token)
    return result

j=[]
topicdict = []
topiclust = []
for i in np.sort(frame['cluster'].unique()):
    processed_docs = frame[frame['cluster']==i]['clean content'].map(preprocess)
    dictionary = gensim.corpora.Dictionary(processed_docs)    
    dictionary.filter_extremes(no_below=1, no_above=0.9, keep_n=100000)    
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]    
    tfidf = models.TfidfModel(bow_corpus)


    #TFIDF for corpus
    corpus_tfidf = tfidf[bow_corpus]

    temtp = []
#Using TFIDF Corpus
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=1, id2word=dictionary, passes=2, workers=4)
    for idx, topic in lda_model_tfidf.print_topics(-1):
        for r in lda_model_tfidf.show_topic(idx, topn=10):
            temtp.append(r[0])
    j.append(i)
    topicdict.append(temtp)
    topiclust.append('Topic of Cluster'+str(i))
        

topicmodel = {'Cluster':topiclust, 'Topic': topicdict}

topicdf = pd.DataFrame(topicmodel)




import matplotlib.pyplot as plt


nltkstopwords = nltk.corpus.stopwords.words('english')
morestopwords = ['know', 'https', 'http', 'well', 'said', 'one', 'time', 'people', 'look', 'many', 'ago', 'even', 'much', 'didnt', 'see', 'weve', 'say', 'ive', 'got', 'come', 'like', 'thats', 'ever', 'theyre', 'going', 'dont', 'want', 'rrthe', '\r', 'shall', 'made', 'et.', 'al', 'could','would','might','must','need',
                 'rrrrrr','rr', 'h','b', 'sha','wo','y',"'s","'d","'ll","'t","'m","'re","'ve", 
                 "n't", 'readingrrcoronavirus', 'treatmentrbut', 'recoveryrbookr22.95rview', "image", "reuters", "caption", "breaking", "news", "via", "via image caption", "copy", "copyright", "getty", "nbc", "cnn", "images", "using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown"]

stopwords = nltkstopwords + morestopwords


cont = frame['clean content']
clss = frame['cluster']
names = [] 
sds = []
leng = []
arra = []
ffs = []
ww=0
for i in j:
    arra = []
    names.append('Cluster'+str(i))
    leng.append(len(frame[frame['cluster']==i]['clean content']))
    dff = frame[frame['cluster']==i]['clean content']
    for yy in frame[frame['cluster']==i]['clean content']:
        arra.append(yy)
    ffs.append(arra)
    word_cloud(arra, ww, stopwords)
    ww=ww+1


#Visualize the data
import matplotlib as mpl
from sklearn.manifold import MDS
MDS()
#Calculate distances for plotting

dist = 1 - cosine_similarity(tfidf_matrix)

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
print()
print()

cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#7570b3', 6: '#e7298a'}
#set up cluster names using a dict
cluster_names = {0: 'C1', 
                 1: 'C2', 
                 2: 'C3', 
                 3: 'C4', 
                 4: 'C5',
                 5: 'C6',
                 6: 'C7'}
#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters) )
#group by cluster
groups = df.groupby('label')
# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], size=8)      
plt.show() #show the plot


#-------------------------------------------------------------------------------#
#-------------------------------Lets Train a model------------------------------#
#-------------------------------------------------------------------------------#
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
 
#Test Train Split
X = frame['clean content']
y = frame['cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



tf_vectorizer=CountVectorizer() 
X_train_tf = tf_vectorizer.fit_transform(X_train)
X_test_tf = tf_vectorizer.transform(X_test)



tf_tfidf = TfidfTransformer()
tf_tfidf.fit(X_train_tf)
X_train_tf_idf = tf_tfidf.transform(X_train_tf)

tf_tfidf.fit(X_test_tf)
X_test_tf_idf = tf_tfidf.transform(X_test_tf)

sns.heatmap(X_train_tf_idf.todense()[:,np.random.randint(0,X_train_tf_idf.shape[1],100)]==0, vmin=0, vmax=1, cbar=False).set_title('Sparse Matrix Sample')


from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train_tf_idf, y_train)
predictions = text_classifier.predict(X_test_tf_idf)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))


#---------------------------------------------------------------------------
#------------------------ Cross Validation ---------------------------------
#---------------------------------------------------------------------------
tfidf_vectorizer = TfidfVectorizer(max_df=0.90,
                                min_df=0.1,
                                 use_idf=True, ngram_range=(1,3))
X_t = tfidf_vectorizer.fit_transform(X)
model = classifier.fit(X_t, y)
predicted = model.predict(X_t)
predicted_prob = model.predict_proba(X_t)

scores = cross_val_score(classifier, X_t, y, cv=20)

plt.plot(scores, marker ='*')

classes = np.unique(y)
y_array = pd.get_dummies(y, drop_first=False).values
    
## Accuracy, Precision, Recall

from sklearn import metrics
accuracy = metrics.accuracy_score(y, predicted)
auc = metrics.roc_auc_score(y_array, predicted_prob, 
                            multi_class="ovr")

print("Accuracy:",  round(accuracy,2))
print("Auc:", round(auc,2))
print("Detail:")
print(metrics.classification_report(y, predicted))
    
## Plot confusion matrix
cm = metrics.confusion_matrix(y, predicted)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, 
       yticklabels=classes, title="Confusion matrix")
plt.yticks(rotation=0)

fig, ax = plt.subplots(nrows=1, ncols=2)
## Plot roc
for i in range(len(classes)):
    fpr, tpr, thresholds = metrics.roc_curve(y_array[:,i],  
                           predicted_prob[:,i])
    ax[0].plot(fpr, tpr, lw=3, 
              label='{0} (area={1:0.2f})'.format(classes[i], 
                              metrics.auc(fpr, tpr))
               )
ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05], 
          xlabel='False Positive Rate', 
          ylabel="True Positive Rate (Recall)", 
          title="Receiver operating characteristic")
ax[0].legend(loc="lower right")
ax[0].grid(True)
    
## Plot precision-recall curve
for i in range(len(classes)):
    precision, recall, thresholds = metrics.precision_recall_curve(
                 y_array[:,i], predicted_prob[:,i])
    ax[1].plot(recall, precision, lw=3, 
               label='{0} (area={1:0.2f})'.format(classes[i], 
                                  metrics.auc(recall, precision))
              )
ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', 
          ylabel="Precision", title="Precision-Recall curve")
ax[1].legend(loc="best")
ax[1].grid(True)
plt.show()

