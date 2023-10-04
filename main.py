import os

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

data_path = os.path.join(os.getcwd(), 'data')
results_path = os.path.join(data_path, 'Q_dataset.xlsx')
docs_path = os.path.join(data_path, 'CF', 'docs')
docs_no = 1239


def read_data():
    r = pd.read_excel(results_path)
    d = {}
    # for i in range(1, (docs_no + 1)):
    #     d_path = docs_path + os.sep + str(i).rjust(5, '0')
    #     with open(d_path, 'r') as f:
    #         d[i] = f.read()
    b = r[['MAP_win7', 'Map_Set']].apply(lambda row: 1 if row['MAP_win7'] > row['Map_Set'] else 0, axis=1)
    q = get_embeddings(r['Q_id'].tolist(), r['Q_text'].tolist())
    return q, b, d


def get_embeddings(ids, texts):
    t = []
    for i in range(len(ids)):
        t.append(TaggedDocument(texts[i], [ids[i]]))
    m = Doc2Vec(t)
    m.train(t, total_examples=m.corpus_count, epochs=m.epochs)
    v = []
    for i in ids:
        v.append(m.dv[i])
    return v


if __name__ == '__main__':
    X, y, docs = read_data()
    scoring = ['precision_macro', 'recall_macro']
    clf = SVC()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    # scores = cross_validate(clf, X, y, scoring=scoring)
    # print(scores)
