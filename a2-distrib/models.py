# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor:
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def extract_features(self, sentence: List[str], target_len:int) -> List[int]:
        res = []
        for i in range(target_len-len(sentence)):
            res.append(0)

        for word in sentence:
            id =  self.indexer.index_of(word)
            if id==-1:
                res.append(1)
            else:
                res.append( id )
        return res


def correction_word(word:str, indexer: Indexer, dict):
    import nltk

    if indexer.index_of(word)!=-1:
        return word

    if word in dict.keys():
        return dict[word]
    
    if len(word)<=3: 
        return word


    minDis = 100
    res = ""
    for i in range(indexer.__len__()):
        target = indexer.get_object(i)
        if target[0:3]!=word[0:3]:
            continue
        distance = nltk.edit_distance(word, target)
        if distance<minDis:
            res = target
            minDis = distance

    dict[word] = res
    return res


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, model, feature_extractor: FeatureExtractor):
        self.model = model
        self.feature_extractor = feature_extractor
        self.dict = {}

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        # correction on sentence
        indexer = self.feature_extractor.indexer
        if has_typos:
            for i in range(len(ex_words)):
                ex_words[i] =  correction_word(ex_words[i], indexer, self.dict) 
        
        input = torch.LongTensor(self.feature_extractor.extract_features(ex_words, len(ex_words)))
        logits = self.model.forward(input[None,:])
        return torch.argmax(logits[0]).item()



def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    feature_extractor = FeatureExtractor(word_embeddings.word_indexer)
    model = DAN(word_embeddings, n_hidden=16, n_output=2)
    classifer = NeuralSentimentClassifier(model, feature_extractor)

    num_epochs = 15
    initial_learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    loss = torch.nn.CrossEntropyLoss()
    batch_size = 8
    for epoch in range(0, num_epochs):
        permutation = torch.randperm(len(train_exs))
        for i in range(0, len(train_exs) , batch_size):
            indices = permutation[i:i+batch_size]

            batch_sentence = [ train_exs[i].words for i in indices ]
            batch_label = [ train_exs[i].label for i in indices ]
            
            max_len = 0 
            for words in batch_sentence:
                if len(words)>max_len:
                    max_len = len(words)

            batch_input = []
            for words in batch_sentence:
                batch_input.append( feature_extractor.extract_features(words, max_len) )

            input = torch.LongTensor(batch_input)
            logit = model.forward(input)
            label = torch.tensor(batch_label, dtype=torch.long)
            l = loss(logit, label) 

            optimizer.zero_grad()

            l.backward()
            optimizer.step()

        acc,_,_ = evaluate(classifer, dev_exs, has_typos=False)
        # if acc>0.77:
        #     break
    return classifer
        


class DAN(torch.nn.Module):
    def __init__(self, word_embeddings: WordEmbeddings, n_hidden, n_output):
        super().__init__()
        
        embedding = word_embeddings.get_initialized_embedding_layer()
        embedding.padding_idx = 0
        self.embedding = embedding

        len = word_embeddings.get_embedding_length()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(len, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_output),
        )
    

    def forward(self, input):
        embedded = self.embedding(input)
        embedded_average = torch.mean(embedded, dim=1)
        return self.net(embedded_average)


def evaluate(classifier, exs, has_typos):
    """
    Evaluates a given classifier on the given examples
    :param classifier: classifier to evaluate
    :param exs: the list of SentimentExamples to evaluate on
    :return: None (but prints output)
    """
    return print_evaluation([ex.label for ex in exs], classifier.predict_all([ex.words for ex in exs], has_typos))


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
    acc = float(num_correct) / num_total
    output_str = "Accuracy: %i / %i = %f" % (num_correct, num_total, acc)
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    output_str += ";\nPrecision (fraction of predicted positives that are correct): %i / %i = %f" % (num_pos_correct, num_pred, prec)
    output_str += ";\nRecall (fraction of true positives predicted correctly): %i / %i = %f" % (num_pos_correct, num_gold, rec)
    output_str += ";\nF1 (harmonic mean of precision and recall): %f;\n" % f1
    print(output_str)
    return acc, f1, output_str