'''
from itertools import count
import re
from util import *
import math

# Add your import statements here




class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The precision value as a number between 0 and 1
        """

        precision = -1

        #Fill in code here

        relevant_docs_count = 0

        docs_till_k = query_doc_IDs_ordered[:k]
        for i in docs_till_k:
            if i in true_doc_IDs:
                relevant_docs_count += 1
        
        precision = relevant_docs_count/k

        return precision


    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean precision value as a number between 0 and 1
        """

        meanPrecision = -1
        sum_precision = 0

        #Fill in code here

        #preprocessing the qrels to create the relevant documents list
        #for each query

        # true_docs_all = dict()
        # for i in qrels:
        #     if int(i["query_num"]) not in true_docs_all.keys():
        #         true_docs_all[int(i["query_num"])] = []
            
        #     true_docs_all[int(i["query_num"])].append(int(i['id']))
        true_docs_all = RelDocs(qrels)



        for i in range(len(doc_IDs_ordered)):
            #this loop iterated over all the queries
            q_id = int(query_ids[i])
            predicted_docs = doc_IDs_ordered[i]

            true_docs = true_docs_all[q_id]

            precision = self.queryPrecision(predicted_docs, q_id, true_docs, k)
            sum_precision += precision
        
        meanPrecision = sum_precision/len(query_ids)

        return meanPrecision

    
    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The recall value as a number between 0 and 1
        """

        recall = -1

        #Fill in code here
        relevant_docs_count = 0

        if len(true_doc_IDs) == 0:
            return 0
        
        docs_till_k = query_doc_IDs_ordered[:k]
        for i in docs_till_k:
            if i in true_doc_IDs:
                relevant_docs_count += 1
        
        recall = relevant_docs_count/len(true_doc_IDs)

        return recall


    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean recall value as a number between 0 and 1
        """

        meanRecall = -1

        #Fill in code here
        sum_recall = 0

        #Fill in code here

        #preprocessing the qrels to create the relevant documents list
        #for each query

        # true_docs_all = dict()
        # for i in qrels:
        #     if int(i["query_num"]) not in true_docs_all.keys():
        #         true_docs_all[int(i["query_num"])] = []
            
        #     true_docs_all[int(i["query_num"])].append(int(i['id']))
        true_docs_all = RelDocs(qrels)



        for i in range(len(doc_IDs_ordered)):
            #this loop iterated over all the queries
            q_id = int(query_ids[i])
            predicted_docs = doc_IDs_ordered[i]

            true_docs = true_docs_all[q_id]

            recall = self.queryRecall(predicted_docs, q_id, true_docs, k)
            sum_recall += recall
        
        meanRecall = sum_recall/len(query_ids)




        return meanRecall


    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The fscore value as a number between 0 and 1
        """

        fscore = -1

        #Fill in code here
        precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

        if precision == 0 and recall == 0:
            return 0

        fscore = (2 * precision * recall)/(precision + recall)


        return fscore


    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value
        
        Returns
        -------
        float
            The mean fscore value as a number between 0 and 1
        """

        meanFscore = -1

        #Fill in code here
        sum_fscore = 0

        #Fill in code here

        #preprocessing the qrels to create the relevant documents list
        #for each query

        # true_docs_all = dict()
        # for i in qrels:
        #     if int(i["query_num"]) not in true_docs_all.keys():
        #         true_docs_all[int(i["query_num"])] = []
            
        #     true_docs_all[int(i["query_num"])].append(int(i['id']))
        true_docs_all = RelDocs(qrels)



        for i in range(len(doc_IDs_ordered)):
            #this loop iterated over all the queries
            q_id = int(query_ids[i])
            predicted_docs = doc_IDs_ordered[i]

            true_docs = true_docs_all[q_id]

            fscore = self.queryFscore(predicted_docs, q_id, true_docs, k)
            sum_fscore += fscore
        
        meanFscore = sum_fscore/len(query_ids)




        return meanFscore
    

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, qrels,k):
        """
        Computation of nDCG of the Information Retrieval System
        at given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg5 : int
            The k value

        Returns
        -------
        float
            The nDCG value as a number between 0 and 1
        """

        nDCG = 0

        #read qrels somehow

        #collecting relevance dictionaries for the particular query
        rel_dict = []
        for i in qrels:
            if int(i["query_num"]) == query_id:
                rel_dict.append(i)
        
        relevance_scores = []
        for i in query_doc_IDs_ordered:
            if i not in true_doc_IDs:
                relevance_scores.append(0)
            else:
                for j in rel_dict:
                    if int(j["id"]) == i:
                        relevance_scores.append(5-j["position"])
        

        ideal_relevance_scores = sorted(relevance_scores, reverse=True)
        
        docs_till_k = query_doc_IDs_ordered[:k]
        DCG = 0
        for i in range(len(docs_till_k)):
            DCG += relevance_scores[i]/(math.log2(i+2)) #added 2 here instead of 1 because list indices start from 0
        
        iDCG = 0
        for i in range(len(docs_till_k)):
            iDCG += ideal_relevance_scores[i]/(math.log2(i+2))
        
        nDCG = DCG / iDCG

        return nDCG


    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of nDCG of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean nDCG value as a number between 0 and 1
        """

        meanNDCG = -1

        #Fill in code here
        sum_ndcg = 0

        #Fill in code here

        #preprocessing the qrels to create the relevant documents list
        #for each query

        # true_docs_all = dict()
        # for i in qrels:
        #     if int(i["query_num"]) not in true_docs_all.keys():
        #         true_docs_all[int(i["query_num"])] = []
            
        #     true_docs_all[int(i["query_num"])].append(int(i['id']))
        true_docs_all = RelDocs(qrels)



        for i in range(len(doc_IDs_ordered)):
            #this loop iterated over all the queries
            q_id = int(query_ids[i])
            predicted_docs = doc_IDs_ordered[i]

            true_docs = true_docs_all[q_id]

            ndcg = self.queryNDCG(predicted_docs, q_id, true_docs, qrels, k)
            sum_ndcg += ndcg
        
        meanNDCG = sum_ndcg/len(query_ids)


        return meanNDCG


    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of average precision of the Information Retrieval System
        at a given value of k for a single query (the average of precision@i
        values for i such that the ith document is truly relevant)

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The average precision value as a number between 0 and 1
        """

        avgPrecision = -1
        count_relevant = 0
        sum_precision = 0
        docs_till_k = query_doc_IDs_ordered[:k]
        for i in range(len(docs_till_k)):
            if docs_till_k[i] in true_doc_IDs:
                count_relevant += 1
                sum_precision += (count_relevant)/(i+1)
        
        if count_relevant == 0 :
            return 0
            
        avgPrecision = sum_precision/count_relevant

            

        #Fill in code here

        return avgPrecision


    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of MAP of the Information Retrieval System
        at given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The MAP value as a number between 0 and 1
        """

        meanAveragePrecision = -1

        #Fill in code here
        sum_ap = 0

        #Fill in code here

        #preprocessing the qrels to create the relevant documents list
        #for each query

        # true_docs_all = dict()
        # for i in qrels:
        #     if int(i["query_num"]) not in true_docs_all.keys():
        #         true_docs_all[int(i["query_num"])] = []
            
        #     true_docs_all[int(i["query_num"])].append(int(i['id']))
        true_docs_all = RelDocs(qrels)



        for i in range(len(doc_IDs_ordered)):
            #this loop iterated over all the queries
            q_id = int(query_ids[i])
            predicted_docs = doc_IDs_ordered[i]

            true_docs = true_docs_all[q_id]

            ap = self.queryAveragePrecision(predicted_docs, q_id, true_docs, k)
            sum_ap += ap
        
        meanAveragePrecision = sum_ap/len(query_ids)




        return meanAveragePrecision

'''

import math
from util import RelDocs

class Evaluation:
    """
    Compute various evaluation metrics for an IR system:
    - Precision, Recall, F-score, nDCG, and Average Precision (AP),
      both per-query and averaged (mean) across queries.
    """

    def queryPrecision(self, ranked_docs, query_id, relevant_docs, k):
        """
        Precision@k for a single query.
        """
        top_k = ranked_docs[:k]
        hit_count = sum(1 for doc in top_k if doc in relevant_docs)
        return hit_count / k if k > 0 else 0.0

    def meanPrecision(self, all_ranked, query_ids, qrels, k):
        """
        Mean Precision@k over all queries.
        """
        rel_map = RelDocs(qrels)
        scores = [
            self.queryPrecision(ranked, qid, rel_map[int(qid)], k)
            for ranked, qid in zip(all_ranked, query_ids)
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def queryRecall(self, ranked_docs, query_id, relevant_docs, k):
        """
        Recall@k for a single query.
        """
        if not relevant_docs:
            return 0.0
        top_k = ranked_docs[:k]
        hit_count = sum(1 for doc in top_k if doc in relevant_docs)
        return hit_count / len(relevant_docs)

    def meanRecall(self, all_ranked, query_ids, qrels, k):
        """
        Mean Recall@k over all queries.
        """
        rel_map = RelDocs(qrels)
        scores = [
            self.queryRecall(ranked, qid, rel_map[int(qid)], k)
            for ranked, qid in zip(all_ranked, query_ids)
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def queryFscore(self, ranked_docs, query_id, relevant_docs, k):
        """
        F1-score@k for a single query.
        """
        p = self.queryPrecision(ranked_docs, query_id, relevant_docs, k)
        r = self.queryRecall(ranked_docs, query_id, relevant_docs, k)
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    def meanFscore(self, all_ranked, query_ids, qrels, k):
        """
        Mean F1-score@k over all queries.
        """
        rel_map = RelDocs(qrels)
        scores = [
            self.queryFscore(ranked, qid, rel_map[int(qid)], k)
            for ranked, qid in zip(all_ranked, query_ids)
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def queryNDCG(self, ranked_docs, query_id, relevant_docs, qrels, k):
        """
        nDCG@k for a single query.
        """
        # Gather all relevance entries for this query
        entries = [e for e in qrels if int(e['query_num']) == query_id]
        # Build a relevance score map: higher is more relevant
        score_map = {int(e['id']): (5 - e['position']) for e in entries}
        # Compute gains for top-k
        top_k = ranked_docs[:k]
        gains = [score_map.get(doc, 0) for doc in top_k]
        # Ideal ordering: sort gains descending
        ideal = sorted(gains, reverse=True)
        # DCG and IDCG
        dcg = sum(gain / math.log2(idx + 2) for idx, gain in enumerate(gains))
        idcg = sum(ideal_score / math.log2(idx + 2) for idx, ideal_score in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0

    def meanNDCG(self, all_ranked, query_ids, qrels, k):
        """
        Mean nDCG@k over all queries.
        """
        scores = [
            self.queryNDCG(ranked, int(qid), RelDocs(qrels)[int(qid)], qrels, k)
            for ranked, qid in zip(all_ranked, query_ids)
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def queryAveragePrecision(self, ranked_docs, query_id, relevant_docs, k):
        """
        Average Precision@k for a single query.
        """
        top_k = ranked_docs[:k]
        num_hits = 0
        sum_prec = 0.0
        for idx, doc in enumerate(top_k, start=1):
            if doc in relevant_docs:
                num_hits += 1
                sum_prec += num_hits / idx
        return (sum_prec / num_hits) if num_hits > 0 else 0.0

    def meanAveragePrecision(self, all_ranked, query_ids, qrels, k):
        """
        Mean Average Precision@k over all queries (MAP).
        """
        rel_map = RelDocs(qrels)
        scores = [
            self.queryAveragePrecision(ranked, qid, rel_map[int(qid)], k)
            for ranked, qid in zip(all_ranked, query_ids)
        ]
        return sum(scores) / len(scores) if scores else 0.0


