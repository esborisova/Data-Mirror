import pandas as pd
import gensim
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.corpora import Dictionary

import pyLDAvis.gensim_models
import pyLDAvis.sklearn
#pyLDAvis.enable_notebook()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from gensim.models import CoherenceModel


def compute_coherence_values(dictionary, 
                             corpus, 
                             texts, 
                             max_topics, 
                             min_topics, 
                             step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    
    for num_topics in range(min_topics, max_topics, step):

        model = LdaModel(corpus = corpus, 
                         num_topics = num_topics, 
                         id2word = dictionary,
                         update_every = 1,
                         passes = 10,
                         per_word_topics = True)

        model_list.append(model)
        coherencemodel = CoherenceModel(model = model, 
                                        texts = texts, 
                                        dictionary = dictionary, 
                                        coherence='c_v')

        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values



df1 = pd.read_pickle('prepared_txt/all_titles_da_df.pkl')

bigram = gensim.models.Phrases(df1['lemmatized_titles'])
titles_bigrams = [bigram[line] for line in df1['lemmatized_titles']]

dictionary = Dictionary(titles_bigrams)
corpus = [dictionary.doc2bow(text) for text in titles_bigrams]

min_topics = 2
max_topics = 10
step = 2
x = range(min_topics, max_topics, step)

model_list, coherence_values = compute_coherence_values(dictionary = dictionary, 
                                                        corpus = corpus, 
                                                        texts = titles_bigrams, 
                                                        min_topics = min_topics, 
                                                        max_topics = max_topics, 
                                                        step = step)

best_result_index = coherence_values.index(max(coherence_values))
#print(x[best_result_index])

ldamodel = LdaModel(corpus = corpus, 
                    num_topics = x[best_result_index], 
                    id2word = dictionary,
                    update_every = 1,
                    passes = 10,
                    per_word_topics = True)

model =pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
pyLDAvis.save_html(model, 'models/da/all_titles_da.html')