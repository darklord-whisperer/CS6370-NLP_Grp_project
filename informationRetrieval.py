'''from heapq import merge
from util import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from numpy import dot
from numpy.linalg import norm
import numpy as np


# Add your import statements here




class InformationRetrieval():

    def __init__(self):
        self.index = None
        self.tfidfModel = None

    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        index = None

        #Fill in code here

        #first merging the sentences in each doc to get one big list for each doc
        merged_docs = []
        for i in range(len(docs)):
            temp = []
            for j in docs[i]:
                temp.extend(j)
            merged_docs.append(temp)

        #converting the tokens into sentences for tfidf vectorizer
        final_corpus = []
        for i in range(len(merged_docs)):
            temp = " "
            final_corpus.append(temp.join(merged_docs[i]))

        vectorizer = TfidfVectorizer()
        vectorizer.fit(final_corpus)
        X = vectorizer.transform(final_corpus)
        self.tfidfModel = vectorizer
        doc_term_matrix = X.toarray().tolist()
        index = dict()
        for i in range(len(docIDs)):
            index[docIDs[i]] = doc_term_matrix[i]


        self.index = index


    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query
        

        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        doc_IDs_ordered = []

        #Fill in code here
        #first merging the sentences in each doc to get one big list for each doc
        merged_queries = []
        for i in range(len(queries)):
            temp = []
            for j in queries[i]:
                temp.extend(j)
            merged_queries.append(temp)

        #converting the tokens into sentences for tfidf vectorizer
        final_queries = []
        for i in range(len(merged_queries)):
            temp = " "
            final_queries.append(temp.join(merged_queries[i]))

        vectorizer = self.tfidfModel
        X = vectorizer.transform(final_queries)
        doc_term_matrix = X.toarray().tolist()

        for i in range(len(doc_term_matrix)):
            #here i refers to the query
            #i.e., we are iterating over each query.
            query_vector = doc_term_matrix[i]
            doc_query_sim = dict()
            doc_order = []

            #for each document, calculate cosine similarity with query
            for j in self.index.keys():
                doc_vector = self.index[j]
                cos_sim = cosine_similarity(np.reshape(query_vector, (1,-1)),np.reshape(doc_vector, (1,-1)))
                #cos_sim = dot(query_vector, doc_vector)/(norm(query_vector)*norm(doc_vector))
                doc_query_sim[j] = cos_sim

            sort_orders = sorted(doc_query_sim.items(), key=lambda x: x[1], reverse=True)
            for k in sort_orders:
                doc_order.append(k[0])

            doc_IDs_ordered.append(doc_order)
    
        return doc_IDs_ordered

        '''

from heapq import merge
from util import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class InformationRetrieval:

    def __init__(self):
        # Will hold the TF-IDF vectorizer and a mapping of docIDs to their TF-IDF vectors
        self.tfidfModel = None
        self.index = {}

    def buildIndex(self, docs, docIDs):
        """
        Builds a TF-IDF index for the given documents.

        Parameters
        ----------
        docs : list of list of list of str
            Each document is a list of sentences, and each sentence is a list of tokens.
        docIDs : list of int
            Unique identifiers for each document.
        """
        # Flatten each document's sentences into a single string
        corpus = [
            " ".join(token for sentence in doc for token in sentence)
            for doc in docs
        ]

        # Fit TF-IDF on the document corpus
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Store the vectorizer
        self.tfidfModel = vectorizer

        # Build index: map docID to its TF-IDF vector (dense array)
        for idx, docID in enumerate(docIDs):
            # Convert sparse vector to dense 1D array
            self.index[docID] = tfidf_matrix[idx].toarray().ravel()

    def rank(self, queries):
        """
        Ranks documents for each query by cosine similarity.

        Parameters
        ----------
        queries : list of list of list of str
            Each query is a list of sentences, and each sentence is a list of tokens.

        Returns
        -------
        list of list of int
            For each query, a list of document IDs sorted by decreasing relevance.
        """
        # Flatten each query's sentences into a single string
        query_texts = [
            " ".join(token for sentence in qry for token in sentence)
            for qry in queries
        ]

        # Transform queries into TF-IDF space
        query_matrix = self.tfidfModel.transform(query_texts)

        # Prepare document matrix and ID list in consistent order
        docIDs = list(self.index.keys())
        doc_matrix = np.vstack([self.index[d] for d in docIDs])

        # Compute cosine similarities (queries x documents)
        sim_matrix = cosine_similarity(query_matrix, doc_matrix)

        ranked_results = []
        # For each query, sort document IDs by similarity
        for sims in sim_matrix:
            # argsort gives ascending order; reverse for descending
            sorted_idx = np.argsort(sims)[::-1]
            ranked_results.append([docIDs[i] for i in sorted_idx])

        return ranked_results





