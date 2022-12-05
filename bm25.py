import math
from collections import defaultdict
from bs4 import BeautifulSoup
import json
from porterStemming import PorterStemmer

NUM_OF_DOCS = 100
AVG_LEN_DOCS = 14795.75 #for 100 docs where total length is 1479575
# Code figured out avg document length -> keeping in case we need it again
# doc_length = []
# counter = 1
# with open ('./100Lines.jsonl', encoding="utf-8", errors="ignore") as corpusFile:
#         for line in corpusFile:
#             json_doc = json.loads(line)
#             contents = ' '.join(BeautifulSoup(json_doc['contents'], "html.parser").stripped_strings)
#             doc_length.append(len(contents))
#             if counter % 100 == 0:
#                 print("counter is at " + str(counter)) 
#             counter += 1
#         print(doc_length)

# LEN_DOCS = sum(([doc for doc in doc_length]))
# AVG_LEN_DOCS = LEN_DOCS/NUM_OF_DOCS
# print("avg doc length is " + str(AVG_LEN_DOCS))

stopwords=[]
with open('stopwords.txt', 'r') as f:
    for term in f:
        term = term.split('\n')
        stopwords.append(term[0])

# create a dictionary where key is id and value is the document content
documents = {}
with open ('./100Lines.jsonl', encoding="utf-8", errors="ignore") as corpusFile:
            for line in corpusFile:
                json_doc = json.loads(line)
                docID = json_doc['id']
                contents = ' '.join(BeautifulSoup(json_doc['contents'], "html.parser").stripped_strings)
                contents_list=contents.split()
                filtered_terms = []
                acceptable_characters = set('-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
                for term in contents_list:
                    answer = ''.join(filter(acceptable_characters.__contains__, term))
                    filtered_terms.append(answer)
                documents[docID] = filtered_terms

# inverted index where key is term and value is list of all docid's it's in
inverted_index = defaultdict(set)
for docid, terms in documents.items():
    for term in terms:
        inverted_index[term.lower()].add(docid)

def tf_idf_score(k1, b, term, docid):
    # get term frequency by counting length of inverted index
    tf = len(inverted_index[term.lower()])
    # stem query term and make it lowercase
    p=PorterStemmer()
    stemmed_term_list = (p.stem(term, 0, len(term)-1))
    stemmed_term = ("".join(stemmed_term_list)).lower()
    # get document frequency by counting number of times term appears in given doc
    documents_lower = [x.lower() for x in documents[docid]]
    df = documents_lower.count(stemmed_term)

    idf_comp = math.log((NUM_OF_DOCS - tf + 0.5)/(tf+0.5))
    tf_comp = ((k1 + 1)*df)/(k1*((1-b) + b*(len(list(filter(None, documents[1512303])))/AVG_LEN_DOCS))+df)
    return idf_comp * tf_comp

# creates a matrix for all terms
def create_tf_idf(k1, b):
    tf_idf = defaultdict(dict)
    for term in set(inverted_index.keys()):
        for docid in inverted_index[term]:
            tf_idf[term][docid] = tf_idf_score(k1, b, term, docid)
    return tf_idf

tf_idf = create_tf_idf(1.5, 0.5) # k1 = 1.5, b = 0.5 (default values)

def get_query_tf_comp(k3, term, query_tf):
    return ((k3+1)*query_tf[term])/(k3 + query_tf[term])

def retrieve_docs(query, result_count):
    q_terms = [term.lower() for term in query.split() if term not in stopwords]
    query_tf = {}
    for term in q_terms:
        query_tf[term] = query.get(term, 0) + 1
    
    scores = {}

    for word in query_tf.keys():
        for document in inverted_index[word]:
            scores[document] = scores.get(document, 0) + (tf_idf[word][document]*get_query_tf_comp(0,word,query_tf)) #k3 = 0 (default)
    return sorted(scores.items(), key=lambda x : x[1], reverse=True)[:result_count]

# retrieve similarity score for query for first x documents -> convert this to command line input
retrieve_docs("James Bond", 5)