'''
# Add your import statements here

import nltk
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')




def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

def RelDocs(qrels):
    true_docs_all = dict()

    for i in qrels:
        if int(i["query_num"]) not in true_docs_all.keys():
            true_docs_all[int(i["query_num"])] = []
            
        true_docs_all[int(i["query_num"])].append(int(i['id']))
    return true_docs_all

# Add any utility functions here

'''
# Add your import statements here
import nltk
from nltk.corpus import wordnet
from itertools import chain
import numpy as np
from mediawiki import MediaWiki

nltk.download('averaged_perceptron_tagger')

def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def RelDocs(qrels):
    true_docs_all = dict()
    for i in qrels:
        if int(i["query_num"]) not in true_docs_all.keys():
            true_docs_all[int(i["query_num"])] = []
        true_docs_all[int(i["query_num"])].append(int(i['id']))
    return true_docs_all

# Add missing function for flattening nested lists
def chain_lists(list_of_lists):
    """
    Flattens a list of lists into a single list.
    
    Args:
        list_of_lists (list): A list containing other lists
        
    Returns:
        list: A flattened list
    """
    return list(chain.from_iterable(list_of_lists))

# Calculate document frequency for terms
def docFreqCalc(docs):
    """
    Calculate document frequency for each term in the corpus.
    """
    df = {}
    for doc in docs:
        for term in set(doc):  # Use set to count each term only once per document
            if term in df:
                df[term] += 1
            else:
                df[term] = 1
    return df

# Create TF-IDF representation
def create_tf_idf(docs, df, vocabulary):
    """
    Create TF-IDF representation for documents.
    """
    N = len(docs)
    tfidf_docs = []
    
    for doc in docs:
        doc_tfidf = {}
        for term in set(doc):
            if term in df:
                tf = doc.count(term) / len(doc) if len(doc) > 0 else 0
                idf = np.log(N / df[term]) if df[term] > 0 else 0
                doc_tfidf[term] = tf * idf
        tfidf_docs.append(doc_tfidf)
    
    return tfidf_docs

# Create document-term matrix
def createDocMatrix(num_docs, vocab_size, tfidf_docs, vocabulary):
    """
    Create a document-term matrix from TF-IDF values.
    """
    word_index = {word: idx for idx, word in enumerate(vocabulary)}
    doc_matrix = np.zeros((num_docs, vocab_size))
    
    for doc_idx, doc_tfidf in enumerate(tfidf_docs):
        for term, score in doc_tfidf.items():
            if term in word_index:
                term_idx = word_index[term]
                doc_matrix[doc_idx, term_idx] = score
    
    return doc_matrix, word_index

# Generate query vector
def genQueryVector(query, vocabulary, df, num_docs):
    """
    Generate a vector representation for a query.
    """
    query_vector = np.zeros(len(vocabulary))
    
    for term in set(query):
        if term in vocabulary:
            term_idx = vocabulary.index(term)
            tf = query.count(term) / len(query) if len(query) > 0 else 0
            idf = np.log(num_docs / df.get(term, 1)) if term in df else 0
            query_vector[term_idx] = tf * idf
    
    return query_vector

# Fetch Wikipedia docs
def wikipediaDocs(titles):
    """
    Fetch document content from Wikipedia based on titles.
    """
    wikipedia = MediaWiki()
    docs = []
    
    for title in titles:
        try:
            page = wikipedia.page(title)
            docs.append(page.content)
        except:
            docs.append("")
    
    return docs

# Generate ESA vector
def get_ESA_vector(doc, tfidf_wiki, docs_wiki, word_index, docs, word_index_wiki):
    """
    Generate ESA (Explicit Semantic Analysis) vector for a document.
    """
    esa_vector = np.zeros(docs_wiki.shape[0])
    
    for term in doc:
        if term in word_index_wiki:
            term_idx = word_index_wiki[term]
            for i in range(docs_wiki.shape[0]):
                if docs_wiki[i, term_idx] > 0:
                    esa_vector[i] += docs_wiki[i, term_idx]
    
    norm = np.linalg.norm(esa_vector)
    if norm > 0:
        esa_vector = esa_vector / norm
    
    return esa_vector

# Calculate cosine similarity
def return_cosine_similarity(doc_vectors, query_vectors):
    """
    Calculate cosine similarity between document and query vectors.
    """
    similarities = np.zeros((doc_vectors.shape[0], query_vectors.shape[0]))
    
    for i in range(doc_vectors.shape[0]):
        for j in range(query_vectors.shape[0]):
            dot_product = np.dot(doc_vectors[i], query_vectors[j])
            doc_mag = np.linalg.norm(doc_vectors[i])
            query_mag = np.linalg.norm(query_vectors[j])
            if doc_mag > 0 and query_mag > 0:
                similarities[i, j] = dot_product / (doc_mag * query_mag)
    
    return similarities

# Return documents ordered by similarity
def return_orderded_docs(similarity_matrix):
    """
    Return document IDs ordered by similarity to queries.
    """
    ordered_docs = []
    
    for j in range(similarity_matrix.shape[1]):
        query_scores = similarity_matrix[:, j]
        sorted_indices = np.argsort(-query_scores)
        sorted_ids = sorted_indices + 1
        ordered_docs.append(sorted_ids.tolist())
    
    return ordered_docs
