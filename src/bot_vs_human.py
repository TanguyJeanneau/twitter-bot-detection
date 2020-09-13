import pickle
import json
import re
from xml.dom.minidom import parse, parseString

import numpy as np
import scipy.sparse as sp

from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords


def createCNN():
    model = Sequential() 
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(5, 1), activation='relu', input_shape=(100,50,1)))
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(5, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    return model


def createCNNMF():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(2, 5), strides=(2, 1), activation='relu', input_shape=(100,50,1)))
    model.add(Conv2D(64, kernel_size=(2, 5), strides=(2, 1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(2, 5), strides=(2, 1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(2, 5), strides=(2, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='softmax'))
    model.add(Dense(2, activation='softmax'))
    return model


def trainCNN(model, Xtrain, Ytrain, Xdev, Ydev, n_epochs=10):
    model.compile(optimizer=Adam(lr=0.0002), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Xtrain, Ytrain, validation_data=(Xdev, Ydev), epochs=n_epochs)
    return model


def predictCNN(model, Xdev, Ydev):
    Ydev2 = [k[0] for k in Ydev]
    Ypreddev = model.predict(Xdev)
    Ypreddev2 = [k[0]>0.5 for k in Ypreddev]
    accuracy = accuracy_score(Ydev2, Ypreddev2)
    results = classification_report(Ydev2, Ypreddev2)
    print(results)
    print('    ACCURACY: {}'.format(accuracy))
    return results, accuracy


def parseTweets(path='../data/en/', train=True):
    """
    output
    :X: list. list of tweets as strings
    :Y: list. list of the labels
    :Xid: list. list of the user id
    """
    X, Y, idX = [], [], []
    corpus = {}
    truthfile = path+'truth-train.txt' if train else path+'truth-dev.txt'
    for id, l1, l2 in [line.strip().split(':::') for line in open(truthfile)]:
        corpus[id] = (l1 == 'human')
    for userid in corpus.keys():
        Xuser = []
        dom = parse('../data/en/{}.xml'.format(userid))
        doc = dom.getElementsByTagName("documents")[0]
        tweetlist = doc.getElementsByTagName("document")
        for tweet in tweetlist:
            tweettext = tweet.firstChild.wholeText
            Xuser.append(tweettext)
        X.append(Xuser)
        Y.append(corpus[userid])
        idX.append(userid)
    return X, Y, idX


def parseTweetsMF(path='../data/en/', train=True):
    """
    output
    :X: list. list of tweets as strings
    :Y: list. list of the labels
    :Xid: list. list of the user id
    """
    X, Y, idX = [], [], []
    Xmf, Ymf, idXmf = [], [], []
    corpus = {}
    truthfile = path+'truth-train.txt' if train else path+'truth-dev.txt'
    for id, l1, l2 in [line.strip().split(':::') for line in open(truthfile)]:
        corpus[id] = [int(l1 == 'human'), int(l2 == 'male')]
    for userid in corpus.keys():
        Xuser = []
        dom = parse('../data/en/{}.xml'.format(userid))
        doc = dom.getElementsByTagName("documents")[0]
        tweetlist = doc.getElementsByTagName("document")
        for tweet in tweetlist:
            tweettext = tweet.firstChild.wholeText
            Xuser.append(tweettext)
        X.append(Xuser)
        Y.append(corpus[userid][0])
        idX.append(userid)
        if corpus[userid][0]:
            Xmf.append(Xuser)
            Ymf.append(corpus[userid][1])
            idXmf.append(userid)
    return X, Y, idX, Xmf, Ymf, idXmf


def parseTweetsALL(path='../data/en/', train=True):
    """
    output
    :X: list. list of tweets as strings
    :Y: list. list of the labels
    :Xid: list. list of the user id
    """
    Xall, Yall, idX = [], [], []
    corpus = {}
    truthfile = path+'truth-train.txt' if train else path+'truth-dev.txt'
    for id, l1, l2 in [line.strip().split(':::') for line in open(truthfile)]:
        corpus[id] = [int(l1 == 'human'), int(l2 == 'male')]
    for userid in corpus.keys():
        Xuser = []
        dom = parse('../data/en/{}.xml'.format(userid))
        doc = dom.getElementsByTagName("documents")[0]
        tweetlist = doc.getElementsByTagName("document")
        for tweet in tweetlist:
            tweettext = tweet.firstChild.wholeText
            Xuser.append(tweettext)
        Xall.append(Xuser)
        Yall.append(corpus[userid])
        idX.append(userid)
    return np.array(Xall), np.array(Yall), idX


def mytokenizer(string):
    """
    tweet tokenizer
    input
    :string: str. tweet to tokenize.
    output
    :tokenized: list. List of the elements of the tweet, tokenized
    """
    rgx = re.compile(r"""
    (?:(?:[A-Z]{1,2}\.)+)                        # acronymes.
    |(?:@[\w_]+)                                # twitter usernames.
    |(?:\#+[\w_]+[\w\'_\-]*[\w_]+)              # twitter hashtags.
    |(?:[\w_.\-]+@[\w_.\-]+)                    # email addresses.
    |(?:https?://[\w_.\-/~]+)                   # websites.
    |(?:\d{1,2}\ de\ [\w]+\ de\ \d{4})          # full name dates.
    |(?:\d{2}[/|-]\d{2}(?:[/|-]\d{4})?)         # short format dates.
    |(?:(?:[\w]+)(?:[\w'\-_]+)+(?:[\w]+))     # all words with apostrophes and dashes of more than 2 letters.
    |(?:[+\-]?\d+[,/.:-]\d+[+\-]?)              # numbers, including fractions, decimals.
    |(?:[\w_]+)                                 # words without apostrophes or dashes.
    |(?:\.(?:\s*\.){1,})                        # ellipsis dots.
    |(?:\S)                                     # everything else that isn't whitespace
    """, re.I|re.U|re.X)
    tokenized = rgx.findall(string)
    return tokenized


def getLex(lexname):
    pos, neg = set(), set()
    with open(lexname, 'r') as f:
        lexlines = f.readlines()
    for line in lexlines:
        tmp = line.split()
        if line[0] != '#' and len(tmp) == 2:
            if tmp[1] == "positive":
                pos.add(tmp[0])
            elif tmp[1] == "negative":
                neg.add(tmp[0])
    return pos, neg


def addPolarity(X, pos, neg):
    counts = np.zeros((len(X),2), dtype=int)
    for i in range(len(X)):
        toks = mytokenizer(X[i])
        for tok in toks:
            if tok in pos:
                counts[i,0] += 1
            if tok in neg:
                counts[i,1] += 1
    return sp.csr_matrix(counts)


def process(xmlfilename, vectorizer, pos, neg, istrain=False):
    # Preprocessing
    X, Y, Xid = parseTweets(xmlfilename)
    # Vectorization
    if istrain:
        Xvec = vectorizer.fit_transform(X)
    else:
        Xvec = vectorizer.transform(X)
    # Adding polarity information
    Xvec = sp.hstack([Xvec,addPolarity(X, pos, neg)])
    return Xvec, Y, Xid


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def average_word_embedding0(tweet, word_to_vec_map):
    tokens = mytokenizer(tweet.lower())
    avg = np.zeros((50,))
    for w in tokens:
        try:
            avg += word_to_vec_map[w]
        except KeyError:
            0
    avg = avg / len(tokens)
    return avg


def average_word_embedding1(tweet, word_to_vec_map):
    tokens = mytokenizer(tweet.lower())
    avg = np.zeros((54,))
    for w in tokens:
        try:
            avg += np.pad(word_to_vec_map[w], (0, 4), 'constant')
        except KeyError:
            if w[0] == '@':  # User's mention
                avg[50] += 1
            elif w[0] == '#':  # Hashtag
                avg[51] += 1
            elif 'http' in w:  # URL
                avg[52] += 1
            else:
                avg[53] += 1  # Other
    avg = avg / len(tokens)
    return avg


def average_word_embedding2(tweet, word_to_vec_map):
    tokens = mytokenizer(tweet.lower())
    avg = np.zeros((50,))
    for w in tokens:
        try:
            avg += word_to_vec_map[w]
        except KeyError:
            if w[0] == '@':  # User's mention
                avg += word_to_vec_map['<user>']
            elif w[0] == '#':  # Hashtag
                avg += word_to_vec_map['<hashtag>']
            elif 'http' in w:  # URL
                avg += word_to_vec_map['<url>']
    avg = avg / len(tokens)
    return avg


def Xembedding0(X, word_to_vec_map):
    Xemb = []
    for i in range(len(X)):
        x = []
        for tweet in X[i]:
            x.extend(average_word_embedding0(tweet, word_to_vec_map))
        Xemb.append(x)
        if not i%100:
            print(i)
    return Xemb


def Xembedding1(X, word_to_vec_map):
    Xemb = np.empty([len(X), 100, 50])
    for i in range(len(X)):
        x = np.empty([100, 50])
        for j in range(100):
            x[j] = average_word_embedding2(X[i][j], word_to_vec_map)
        Xemb[i] = x
        if not i%100:
            print(i)
    Xemb = Xemb.reshape((len(Xemb), 100, 50, 1))
    return Xemb


def manualgridsearch(X, Y, parameters):
    results = {}
    for m in parameters.keys():
        model = parameters[m]['estimator']
        params = parameters[m]['parameters']


def gridsearch(X, Y, parameters, cv=5, verbose=2, n_jobs=-1):
    results = {}
    for m in parameters.keys():
        model = parameters[m]['estimator']
        params = parameters[m]['parameters']
        clf = GridSearchCV(model, params, cv=cv, verbose=verbose, n_jobs=n_jobs)
        print('    fitting {}...'.format(m))
        clf.fit(X, Y)
        results[m] = {'cv_results_': listify(clf.cv_results_),
                     # 'best_estimator_': clf.best_estimator_,
                      'best_score_': clf.best_score_,
                      'best_params_': clf.best_params_}
        print(results)
        with open('../../DiscoW/ALC/BvH/results.json', 'w') as f:
            json.dump(results, f)
    return results


def listify(dic):
    for k in dic.keys():
        if ((type(dic[k]) == np.ndarray) |
            (type(dic[k]) == np.ma.core.MaskedArray)):
            dic[k] = dic[k].tolist()
    return dic


def predict(model, Xdev, Ydev):
    Ypreddev = model.predict(Xdev)
    accuracy = accuracy_score(Ydev, Ypreddev)
    results = classification_report(Ydev, Ypreddev)
    print(results)
    print('    ACCURACY: {}'.format(accuracy))
    return results, accuracy


def evaluateALL(modelBH, modelMF, Xall, Yall):
    Ybhpred = modelBH.predict(Xall)
    Ybh = [k[0] for k in Yall]
    Ybhpred = [int(k[1]>0.5) for k in Ybhpred]
    print('XXXXXXXXXXXXXXXXXXXXX')
    print('Bot vs Human:')
    print(classification_report(Ybh, Ybhpred))
    print('    ACCURACY: {}'.format(accuracy_score(Ybh, Ybhpred)))
    print('XXXXXXXXXXXXXXXXXXXXX')
    Xmf = np.array([Xall[i] for i in range(len(Xall)) if Yall[i][0]])
    Ymfpred = modelMF.predict(Xmf)
    Ymf = np.array([k[1] for k in Yall if k[0]])
    Ymfpred = [int(k[1]>0.5) for k in Ymfpred]
    print('Male vs Female:')
    print(classification_report(Ymf, Ymfpred))
    print('    ACCURACY: {}'.format(accuracy_score(Ymf, Ymfpred)))
    print('XXXXXXXXXXXXXXXXXXXXX')
    Ymfallpred = modelMF.predict(Xall)
    Ymfallpred = [int(k[1]>0.5) for k in Ymfallpred]
    Yallpred = np.array([[Ybhpred[i], Ymfallpred[i]] if Ybhpred[i] else [Ybhpred[i], 0] for i in range(len(Xall))])
    print('Altogether:')
    print(classification_report(Yall, Yallpred))
    print('    ACCURACY: {}'.format(accuracy_score(Yall, Yallpred)))


def predictALL(modelBH, modelMF, Xall):
    Ybhpred = modelBH.predict(Xall)
    Ybhpred = [int(k[1]>0.5) for k in Ybhpred]
    Ymfallpred = modelMF.predict(Xall)
    Ymfallpred = [int(k[1]>0.5) for k in Ymfallpred]
    Yallpred = np.array([[Ybhpred[i], Ymfallpred[i]] if Ybhpred[i] else [Ybhpred[i], 0] for i in range(len(Xall))])
    return Yallpred


def formatResults(Yallpred, idX, savepath="."):
    if len(Yallpred) != len(idX):
        print('mismatching lengths:')
        print('{} != {}'.format(len(Yallpred), len(idX)))
        return False
    basicfile = r"""<author id="{}"
    lang="{}"
    type="{}"
    gender="{}"
/>"""
    for i in range(len(idX)):
        idi = idX[i]
        langi = "en"
        typei = "human" if Yallpred[i][0] else "bot"
        genderi = "bot" if not Yallpred[i][0] else "male" if Yallpred[i][1] else "female"
        filei = basicfile.format(idi, langi, typei, genderi)
        open("{}/{}_JEANNEAU.xml".format(savepath, idi), "w").write(filei)
        if not i%100:
            print(i)

if __name__ == "__main__":
    do_parse = True
    do_vectorization_again = False
    import_vectorization = False
    import_word_embedding = True
    get_Xs_embeddings0 = False
    get_Xs_embeddings1 = True
    do_classification = False
    model0 = False
    model1 = False
    modelcnn = True
    predictonexternalmodel = True
    modelBHname = 'modelBH.h5'
    modelMFname = 'modelMF.h5'

    if do_parse:
        print('parsing...')
        print('    TRAIN...')
        Xalltrain, Yalltrain, _ = parseTweetsALL(train=True)
        # Xtrain, Ytrain, _, Xtrainmf, Ytrainmf, _ = parseTweetsMF(train=True)
        # Xtrain, Ytrain, _ = parseTweetsMF(train=True)
        print('    DEV...')
        Xalldev, Yalldev, idXdev = parseTweetsALL(train=False)
        # Xdev, Ydev, _, Xdevmf, Ydevmf, _ = parseTweetsMF(train=False)
        # Xdev, Ydev, _ = parseTweetsMF(train=False)
    if do_vectorization_again:
        print('vectorizing...')
        tokenizer = TweetTokenizer()
        vectorizer = CountVectorizer(analyzer='char_wb',
                                     ngram_range=(1,7),
                                     stop_words=stopwords.words('english'),
                                     tokenizer=tokenizer.tokenize)
        print('    flattening...')
        flattenedX = [tweet for usermatrix in X for tweet in usermatrix]
        print('    fitting...')
        vectorizer.fit(flattenedX)
        print('    transforming...')
        Xvec = [vectorizer.transform(usermatrix) for usermatrix in X]
    if import_vectorization:
        print('opening Xvec pickle ...')
        pickle_in = open("Xvec.pickle","rb")
        Xvec = pickle.load(pickle_in)
    if import_word_embedding:
        print('getting word embedding...')
        words_to_index, index_to_words, word_to_vec_map = read_glove_vecs('glove.twitter.27B.50d.txt')
    if get_Xs_embeddings0:
        print('word embedding type 0...')
        print('    TRAIN...')
        Xembtrain = Xembedding0(Xtrain, word_to_vec_map)
        print('    DEV...')
        Xembdev = Xembedding0(Xdev, word_to_vec_map)
    if get_Xs_embeddings1:
        print('word embedding type 1...')
        print('    TRAIN...')
        Xallembtrain = Xembedding1(Xalltrain, word_to_vec_map)
        Xbhembtrain = Xallembtrain.copy()
        Ybhembtrain = to_categorical([k[0] for k in Yalltrain])
        Xmfembtrain = np.array([Xallembtrain[i] for i in range(len(Xallembtrain)) if Yalltrain[i][0]])
        Ymfembtrain = to_categorical([Yalltrain[i][1] for i in range(len(Xallembtrain)) if Yalltrain[i][0]])
        # Xembtrain = Xembedding1(Xtrain, word_to_vec_map)
        # Yembtrain = to_categorical(Ytrain)
        # Xembtrainmf = Xembedding1(Xtrainmf, word_to_vec_map)
        # Yembtrainmf = to_categorical(Ytrainmf)
        print('    DEV...')
        Xallembdev= Xembedding1(Xalldev, word_to_vec_map)
        Xbhembdev = Xallembdev.copy()
        Ybhembdev = to_categorical([k[0] for k in Yalldev])
        Xmfembdev = np.array([Xallembdev[i] for i in range(len(Xallembdev)) if Yalldev[i][0]])
        Ymfembdev = to_categorical([Yalldev[i][1] for i in range(len(Xallembdev)) if Yalldev[i][0]])
        # Xembdev = Xembedding1(Xdev, word_to_vec_map)
        # Yembdev = to_categorical(Ydev)
        # Xembdevmf = Xembedding1(Xdevmf, word_to_vec_map)
        # Yembdevmf = to_categorical(Ydevmf)
    if do_classification:
        print('gridsearching...')
        parameters = {'svm': 
                         {'estimator': SVC(random_state=42),
                          'parameters': 
                             {'kernel': ['linear', 'poly', 'rbf'],
                              'C': [0.001, 0.1, 0.5, 1, 3, 10, 100]}},
                      'rf':
                         {'estimator': RandomForestClassifier(random_state=42),
                          'parameters':
                             {'max_depth': [4, 8],
                              'n_estimators': [200, 800],
                              'max_features': ['sqrt', 0.33]}},
                      'gb': 
                         {'estimator': GradientBoostingClassifier(random_state=42),
                          'parameters':
                             {'max_depth': [4, 8],
                              'n_estimators': [100, 500],
                              'learning_rate': [0.1, 0.2]}}}
        results = gridsearch(Xembtrain, Ytrain, parameters, n_jobs=None, cv=None)
    if model0:
        print('fitting model 0...')
        print('    SVC kernel="linear", C=1')
        svm1 = SVC(C=1, kernel='linear', random_state=42)
        svm1.fit(Xembtrain, Ytrain)
        ressvm1, accsvm1 = predict(svm1, Xembdev, Ydev)
        with open('results_svm.json', 'w') as f:
            json.dump({'report': ressvm1, 'accuracy': accsvm1}, f)
    if model1:
        print('fitting model 1...')
        print('    RF max_depth=4, n_estimators=1000')
        rf1 = RandomForestClassifier(max_depth=4, n_estimators=1000, random_state=42)
        rf1.fit(Xembtrain, Ytrain)
        resrf1, accrf1 = predict(rf1, Xembdev, Ydev)
        with open('results_rf.json', 'w') as f:
            json.dump({'report': resrf1, 'accuracy': accrf1}, f)
    if modelcnn:
        print('FIRST PART : Bot vs Human')
        print('creating CNN...')
        cnn = createCNN()
        print(cnn.summary())
        print('training CNN...')
        cnn = trainCNN(cnn, Xbhembtrain, Ybhembtrain, Xbhembdev, Ybhembdev, n_epochs=9)
        print('Calculating prediction...')
        rescnn, acccnn = predictCNN(cnn, Xbhembdev, Ybhembdev)
        print('SECOND PART : Male vs female')
        print('creating CNN...')
        cnn2 = createCNNMF()
        print(cnn2.summary())
        print('training CNN...')
        cnn2 = trainCNN(cnn2, Xmfembtrain, Ymfembtrain, Xmfembdev, Ymfembdev, n_epochs=9)
        print('Calculating prediction...')
        rescnn2, acccnn2 = predictCNN(cnn2, Xmfembdev, Ymfembdev)
    if predictonexternalmodel:
        modelBH = load_model(modelBHname)
        modelMF = load_model(modelMFname)
        Yallpreddev = predictALL(modelBH, modelMF, Xallembdev)
        evaluateALL(modelBH, modelMF, Xallembdev, Yalldev)
        formatResults(Yallpreddev, idXdev, savepath='./results_JEANNEAU')
        # print('predictions on dev...')
        # for m in results.keys():
        #     Ypreddev = results[m]['best_estimator_'].predict(Xembdev)
        #     results[m]['best_accuracy'] = accuracy_score(Ydev, Ypreddev)
        #     results[m]['classification_report'] = classification_report(Ydev, Ypreddev)
        #     print('    {}:'.format(m))
        #     print(classification_report(Ydev, Ypreddev))
        #     print('    ACCURACY: {}'.format(accuracy_score(Ydev, Ypreddev)))
        # with open('../../DiscoW/ALC/BvH/results_withmetrics.pickle', 'wb') as f:
        #     pickle.dump(results, f)
    print('ok')
