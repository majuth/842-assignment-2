import json
from collections import defaultdict
import math
import pandas as pd

stopwords=[]
with open('stopwords.txt', 'r') as f:
    for term in f:
        term = term.split('\n')
        stopwords.append(term[0])

with open('movies_metadata.json', 'r') as f:
  corpus = json.load(f)

doc_contents = []
doc_titles = []
doc_ids = []

for key in corpus.keys():
    doc_titles.append(corpus[key]['title'])
    doc_contents.append(corpus[key]['overview'])
    doc_ids.append(corpus[key]['id'])

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words={'english'})
X = vectorizer.fit_transform(doc_contents)

true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
model.fit(X)
labels=model.labels_
docs_cl=pd.DataFrame(list(zip(doc_titles,labels)),columns=['title','cluster'])

result={'cluster':labels, 'title': doc_titles,'contents':doc_contents, 'docID': doc_ids}
result=pd.DataFrame(result)

pd.set_option("display.max_colwidth", 300)
query = "christmas romance"
Y = vectorizer.transform([query])
prediction = model.predict(Y)
cluster = int(prediction)
print("Query best fits cluster %d" %cluster)
titles = result[result['cluster'] == cluster]['title']
print("There are %d results" %len(titles))
for title in titles:
    id = result[result['title'] == title]['docID']
    print("Doc ID: %s" %id.to_string(index=False))
    print("Title: %s" %title)
    content = result[result['title'] == title]['contents']
    print("Overview: %s" %content.to_string(index=False))
    print("\n")