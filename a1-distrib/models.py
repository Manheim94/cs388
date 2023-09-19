# models.py

from sentiment_data import *
from sentiment_data import List
from utils import *

from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

        import nltk
        from nltk.corpus import stopwords
        
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        c = Counter()

        # filter words
        filtered_sentence = [w for w in sentence if not w.lower() in self.stop_words]

        sentence = filtered_sentence
        for word in sentence:
            if add_to_indexer:
                id = self.indexer.add_and_get_index(word)
                c[id] += 1
            else:
                id = self.indexer.index_of(word)
                if id != -1:
                    c[id] += 1
        return c

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        c = Counter()
        for i in range(len(sentence)):
            if i==len(sentence)-1:
                continue

            word = sentence[i]+"|"+sentence[i+1]

            if add_to_indexer:
                id = self.indexer.add_and_get_index(word)
                c[id] += 1
            else:
                id = self.indexer.index_of(word)
                if id != -1:
                    c[id] += 1
        return c


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

        import nltk
        from nltk.corpus import stopwords
        
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        c = Counter()
 
        # filter words
        filtered_sentence = [w for w in sentence if not w.lower() in self.stop_words]

        sentence = filtered_sentence
        for i in range(len(sentence)):
            #unigram
            word = sentence[i]
            if add_to_indexer:
                id = self.indexer.add_and_get_index(word)
                c[id] += 1
                if c[id]>3:
                    c[id]=3
            else:
                id = self.indexer.index_of(word)
                if id != -1:
                    c[id] += 1
                    if c[id]>3:
                        c[id]=3    

            # bigram
            if i==len(sentence)-1:
                continue
            word = sentence[i]+"|"+sentence[i+1]
            if add_to_indexer:
                id = self.indexer.add_and_get_index(word)
                c[id] += 1
                if c[id]>3:
                    c[id]=3
            else:
                id = self.indexer.index_of(word)
                if id != -1:
                    c[id] += 1
                    if c[id]>3:
                        c[id]=3
                
        return c


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, numOfFeatures: int, feat_extractor: FeatureExtractor):
        import numpy as np
        self.weights = np.zeros(numOfFeatures)
        self.feat_extractor = feat_extractor
        
    def predict(self, sentence: List[str]) -> int:
        c = self.feat_extractor.extract_features(sentence)

        sum = 0
        for key in c.keys():
            sum += c[key]*self.weights[key]
        
        if sum>0:
            return 1
        else:
            return 0

    def train(self, sentence: List[str], label: int, lr = 1):
        c = self.feat_extractor.extract_features(sentence)

        sum = 0
        for key in c.keys():
            sum += c[key]*self.weights[key]
        
        if sum>0 and label==0:
            for key in c.keys():
                self.weights[key] -= lr*c[key]

        elif sum<=0 and label==1:
            for key in c.keys():
                self.weights[key] += lr*c[key]
            

def sigmoid(z: float):
    import math
    return 1.0/(1.0 + math.exp(-z))

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, numOfFeatures: int, feat_extractor: FeatureExtractor):
        import numpy as np
        self.weights = np.zeros(numOfFeatures)
        self.feat_extractor = feat_extractor
        
    def predict(self, sentence: List[str]) -> int:
        c = self.feat_extractor.extract_features(sentence)

        sum = 0
        for key in c.keys():
            sum += c[key]*self.weights[key]
        
        if sum>0:
            return 1
        else:
            return 0
        
    def train(self, sentence: List[str], label: int, lr = 1):
        c = self.feat_extractor.extract_features(sentence)

        sum = 0
        for key in c.keys():
            sum += c[key]*self.weights[key]

        y = 1
        if label==0:
            y=-1
        import numpy as np
        x = np.zeros(len(self.weights))
        for key in c.keys():
            x[key] = c[key]


        gradient = -sigmoid(-y*sum)*y*x
        self.weights -= lr*gradient 


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, dev_exs: List[SentimentExample]) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """

    for point in train_exs:
        sentence = point.words
        feat_extractor.extract_features(sentence, True)
    len = feat_extractor.get_indexer().__len__()

    # training
    classifier = PerceptronClassifier(len, feat_extractor)
    for i in range(10):
        lr = 0
        if i!=0:
            lr = 1.0/i
        import random
        random.shuffle(train_exs)
        for point in train_exs:
            sentence = point.words
            label = point.label
            classifier.train(sentence,label, lr)
        
        exs = dev_exs
        print_evaluation([ex.label for ex in exs], [classifier.predict(ex.words) for ex in exs])
    return classifier


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, dev_exs: List[SentimentExample]) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    for point in train_exs:
        sentence = point.words
        feat_extractor.extract_features(sentence, True)
    len = feat_extractor.get_indexer().__len__()

    # training
    import random

    random.seed(9) 
    classifier = LogisticRegressionClassifier(len, feat_extractor)
    for i in range(100):
        lr = 1
        if i<10:
            lr = 0.1
        else:
            lr = 0.01
        
        random.shuffle(train_exs)
        for point in train_exs:
            sentence = point.words
            label = point.label
            classifier.train(sentence,label, lr)
        
        exs = dev_exs
        res = print_evaluation([ex.label for ex in exs], [classifier.predict(ex.words) for ex in exs])
        if res:
            break
    return classifier




def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor, dev_exs)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor, dev_exs)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model


def print_evaluation(golds: List[int], predictions: List[int]):
    """
    Prints evaluation statistics comparing golds and predictions, each of which is a sequence of 0/1 labels.
    Prints accuracy as well as precision/recall/F1 of the positive class, which can sometimes be informative if either
    the golds or predictions are highly biased.

    :param golds: gold labels
    :param predictions: pred labels
    :return:
    """
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    res = float(num_correct) / num_total
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    print("Precision (fraction of predicted positives that are correct): %i / %i = %f" % (num_pos_correct, num_pred, prec)
          + "; Recall (fraction of true positives predicted correctly): %i / %i = %f" % (num_pos_correct, num_gold, rec)
          + "; F1 (harmonic mean of precision and recall): %f" % f1)
    return res>0.77