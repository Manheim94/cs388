# models.py

import numpy as np
import torch
from transformer import PositionalEncoding
from torch import optim
import random


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, transformer, vocab_index, chunk_size):
        self.transformer = transformer
        self.vocab_index = vocab_index
        self.chunk_size = chunk_size

    def get_next_char_log_probs(self, context):
        if len(context)==0:
            context = " "
        if len(context)>self.chunk_size:
            context = context[-self.chunk_size:]
        input_indexed = np.array([self.vocab_index.index_of(ci) for ci in context])
        input_tensor = torch.LongTensor(input_indexed)

        l = self.transformer(input_tensor)
        res = l.detach().cpu().numpy()
        return res[ len(res)-1 ]

    def get_log_prob_sequence(self, next_chars, context):
        sum = 0
        for char in next_chars:
            dist = self.get_next_char_log_probs(context)
            sum += dist[ self.vocab_index.index_of(char) ]
            context = context + char
        return sum


class Transformer(torch.nn.Module):
    def __init__(self, vocab_size=27, num_positions=20, d_model=64, num_layers=2):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.positionalEncode = PositionalEncoding(d_model, num_positions)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=4*d_model)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.linear = torch.nn.Linear(d_model, vocab_size)
        self.LogSoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        inputs = self.positionalEncode( self.embed(indices) )

    
        mask = torch.triu( torch.ones(len(indices),len(indices))*float('-inf'), diagonal=1)
        o = self.encoder(inputs, mask= mask,is_causal=True)

        return self.LogSoftmax( self.linear(o) )




def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    chunk_size = 20
    model = Transformer(num_positions=chunk_size)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 4
    for t in range(0, num_epochs):
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train_text)-chunk_size,chunk_size)]
        random.shuffle(ex_idxs)
        loss_fcn = torch.nn.NLLLoss()
        count=0
        for ex_idx in ex_idxs:
            input = train_text[ex_idx: ex_idx+chunk_size]
            input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
            input_tensor = torch.LongTensor(input_indexed)

            expect_tensor = input_tensor
            padding = torch.LongTensor([26])
            input_tensor = torch.cat((padding,input_tensor[0:-1]))

            l= model(input_tensor)
            loss = loss_fcn(l, expect_tensor)
            loss_val = loss.detach().cpu().numpy()
   

            model.zero_grad()
            loss.backward()
            optimizer.step()

            count = count+1
            if count%50==0:
                print(loss_val)
                print_evaluation(dev_text, NeuralLanguageModel(model, vocab_index, chunk_size), vocab_index, args.output_bundle_path)
                
    model.eval()
    return NeuralLanguageModel(model, vocab_index, chunk_size)

def print_evaluation(text, lm, vocab_index, output_bundle_path):
    """
    Runs both the sanity check and also runs the language model on the given text and prints three metrics: log
    probability of the text under this model (treating the text as one log sequence), average log probability (the
    previous value divided by sequence length), and perplexity (averaged "branching favor" of the model)
    :param text: the text to evaluate
    :param lm: model to evaluate
    :param output_bundle_path: the path to print the output bundle to, in addition to printing it
    """
    log_prob = lm.get_log_prob_sequence(text, "")
    avg_log_prob = log_prob/len(text)
    perplexity = np.exp(-log_prob / len(text))
    data = {'log_prob': log_prob, 'avg_log_prob': avg_log_prob, 'perplexity': perplexity}
    print("=====Results=====")
    import json
    print(json.dumps(data, indent=2))
