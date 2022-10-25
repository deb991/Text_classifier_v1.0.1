import re

import matplotlib
import numpy as np
import pandas as  pd
from pprint import pprint# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel# spaCy for preprocessing
import spacy# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
#%matplotlib inline
from spacy.cli.download import download
import logging
download(model="en_core_web_sm")

from data_extract import output_format
from local_build.sandbox.ocr import ocr_analysis
from component.config_parsar import read_config

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

log_file = (read_config())

print('log file:\t', log_file)
logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG)
                    #format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("log file validated !!")


def img_data():
    text_img = ocr_analysis.image_read_lang("C:\\Users\\002CSC744\\Downloads\\pdf_use_case\\photo_6109267155661992742_y.jpg")

    print("text_image:\t", text_img)
    return pd.read_json(output_format(text_img))

#print('img_extract:\t', img_extract)
df = pd.read_json("C:\\Users\\002CSC744\\Documents\\My_Projects"
                  "\\JText-classifier_main\\res\\newsgroups.json")
#df = output_format(img_extract)
print(df.target_names.unique())
df.head()

# Convert to list
data = df.content.values.tolist()
# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]
# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]
pprint(data[:1])


def sent_to_words(sentences):
    logger.info('Sent to words initiated .')
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess
              (str(sentence), deacc=True))  #deacc=True removes punctuations

data_words = list(sent_to_words(data))
print(data_words[:1])

# Build the bigram and trigram models
logger.info('Biggram initiated .')
bigram = gensim.models.Phrases\
    (data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])

# Define function for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    logger.info('Remove Stop Words initiated .')
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    logger.info('make Biggrams initiated .')
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    logger.info('Trigrams initiated .')
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    logger.info('Lemmatization initiated .')
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams

data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization\
    (data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1])

#Readable format of corpus can be obtained by executing below code block.
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Print the keyword of topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))
# a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel\
    (model=lda_model, texts=data_lemmatized,
     dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis

