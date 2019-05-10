import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier


def tokens(doc):
    return (tok.lower() for tok in re.findall(r"\w+", doc))


def frequency(tokens):
    tf_dictionary = {}

    for word in tokens:
        if word in tf_dictionary:
            tf_dictionary[word] += 1
        else:
            tf_dictionary[word] = 1

    return tf_dictionary


def tokens_frequency(doc):
    return frequency(tokens(doc))


def read_all_documents(root):
    labels = []
    docs = []
    for r, dirs, files in os.walk(root):
        for file in files:
            with open(os.path.join(r, file), "r") as f:
                new_header = f.readline()    # First line = news header (Remove that)
                docs.append(f.read().replace(new_header, ""))
            labels.append(r.replace(root, '').strip("\\"))
    return dict([('docs', docs), ('labels', labels)])


def train_model(train_path, tfid, num_neighbors=5):
    data = read_all_documents(train_path)
    documents = data['docs']
    labels = data['labels']

    x_train = tfid.fit_transform(documents)
    y_train = labels

    clf = KNeighborsClassifier(n_neighbors=num_neighbors)
    clf.fit(x_train, y_train)

    return clf


def test_model(test_path, clf, tfid):
    test = read_all_documents(test_path)

    x_test = tfid.transform(test['docs'])
    y_test = test['labels']
    pred = clf.predict(x_test)
    accuracy = clf.score(x_test, y_test)

    return [pred, accuracy]


def main():
    train_path = os.path.join("..", "train")
    test_path = os.path.join("..", "test")

    tfid = TfidfVectorizer(stop_words="english")

    # TRAINING
    clf = train_model(train_path, tfid, 10)

    # TESTING
    prediction = test_model(test_path, clf, tfid)

    # SHOW RESULTS
    print(prediction[0])
    print('TEST accuracy score %f' % (prediction[1]))


if __name__ == '__main__':
    main()
