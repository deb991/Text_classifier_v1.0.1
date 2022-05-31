import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from dagster import job, op, get_dagster_logger


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not " \
       "my father."
doc2 = "My father spends a lot of time driving my sister around to " \
       "dance practice."
doc3 = "Doctors suggest that driving may cause increased stress " \
       "and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but " \
       "my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."

all_doc = [doc1, doc2, doc3, doc4, doc5]

print('Type of all_doc:\t{}, \nall_model '
      'contents of :\t{}'.format(type(all_doc), all_doc))

@op
def clean(doc):
    stop_word_removal = "".join(
        [i for i in doc.lower().split() if i not in stop])
    punctuation_removal = "".join(
        ch for ch in stop_word_removal if ch not in exclude)
    normalize = "".join(
        lemma.lemmatize(word) for word in punctuation_removal.split())
    # print("Stop_word_removal:\t{}, \nPunctuation_removal:\t{},
    # \nNormalize:\t".format(stop_word_removal, punctuation_removal, normalize))

    return normalize


@op(

)
def get_doc_clean(arg):
    doc_clean = [arg(doc).split() for doc in all_doc]

    return doc_clean


@op
def get_term_dictionary(arg):
    # Creating the term dictionary of our corpus, where every
    # unique term is assign to an index
    term_dict = corpora.Dictionary(arg)

    return term_dict


@op
def get_doc_term_matrix(arg1, arg2):
    # converting list of documents(corpus) into Document term matrix
    # using Dictionary prepared above
    doc_term_matrix = [arg1.doc2bow(doc) for doc in arg2]

    return doc_term_matrix


@op
def get_lda_model_output(arg1, arg2):
    # print('Test--1, \nterm_dict:\t{},
    # \ndoc_term_matrix:\t{}'.format(term_dict, doc_term_matrix))

    # Running LDA model

    # Creating LDA object from gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Running & Training LDA model on the document term matrix
    lda_model = Lda(arg1, num_topics=3, id2word=arg2, passes=50)

    return lda_model


@job
def get_materealize():
    #clean()
    #get_doc_clean(clean())
    #get_term_dictionary()
    #get_doc_term_matrix()
    get_lda_model_output(
        get_doc_term_matrix(get_term_dictionary(
            get_doc_clean(clean())),
                            get_doc_clean(clean())),
        get_term_dictionary(get_doc_clean(clean())))

#print('Test--2: \nLDA_Model:\t',
#      lda_model.print_topics(num_topics=3, num_words=3))
