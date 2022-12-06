from kmeans import *
from sklearn.feature_extraction.text import TfidfVectorizer
import json

with open('movies_metadata.json', 'r') as f:
  corpus = json.load(f)

doc_contents = []
doc_titles = []
doc_ids = []

for key in corpus.keys():
    doc_titles.append(corpus[key]['title'])
    doc_contents.append(corpus[key]['overview'])
    doc_ids.append(corpus[key]['id'])

vectorizer = TfidfVectorizer(input=doc_contents, stop_words={'english'})
X = vectorizer.fit_transform(doc_contents)

kmeans = cluster.KMeans(n_clusters=3, init='k-means++')
kmeans.fit(X)

labels=kmeans.labels_
docs_cl=pd.DataFrame(list(zip(doc_titles,labels)),columns=['title','cluster'])
result={'cluster':labels, 'title': doc_titles,'contents':doc_contents, 'docID': doc_ids}
result=pd.DataFrame(result)

query = "christmas romance"
vectorizer = TfidfVectorizer(input=query)
Y = vectorizer.fit_transform([query])

# rest doesn't work bc .predict gives error
# prediction = kmeans.predict(Y)

# cluster = int(prediction)
# print("Query best fits cluster %d" %cluster)
# titles = result[result['cluster'] == cluster]['title']
# print("There are %d results" %len(titles))
# for title in titles:
#     id = result[result['title'] == title]['docID']
#     print("Doc ID: %s" %id.to_string(index=False))
#     print("Title: %s" %title)
#     content = result[result['title'] == title]['contents']
#     print("Overview: %s" %content.to_string(index=False))
#     print("\n")