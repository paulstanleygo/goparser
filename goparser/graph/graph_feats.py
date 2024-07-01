"""This module contains functions for extracting
features for a graph-based neural parser.

The 1-order-atomic feedforward features are from Pei, Ge, and Chang (2015).

The BiLSTM features are from Kiperwasser and Goldberg (2016).
"""
from collections.abc import Callable

from nn_features import NNFeatureMap, NNFeatures
from utils import NULL, UNK, Sentence, Treebank


MAX_DISTANCE = '_MAX_DISTANCE_'
"""String constant representing the index in the deprel_map for the maximum distance
between the head and dependent."""


NNGFeatureExtractor = Callable[[Sentence, int, int], NNFeatures]
"""Type alias for a graph-based neural network feature extractor function that returns
an NNFeatures object given a Sentence list of Tokens."""


def index_nngfeatures(treebank: Treebank) -> NNFeatureMap:
    """Returns an NNFeatureMap object that maps graph features to index given a Treebank.
    
    Args:
        treebank: A Treebank list containing a treebank.
    
    Returns:
        An NNFeatureMap object with feature-to-index maps.
    """
    feat_map = NNFeatureMap()
    
    max_distance = 0
    
    feat_map.word_map[NULL] = 0
    feat_map.pos_map[NULL] = 0

    feat_map.word_map[UNK] = 1
    feat_map.pos_map[UNK] = 1

    for sent in treebank:
        for tok in sent[1:]:
            if tok.form not in feat_map.word_map:
                word_index = len(feat_map.word_map)
                feat_map.word_map[tok.form] = word_index
                feat_map.wordfreq_map[word_index] = 1
            else:
                feat_map.wordfreq_map[feat_map.word_map[tok.form]] += 1
            
            if tok.upos not in feat_map.pos_map:
                feat_map.pos_map[tok.upos] = len(feat_map.pos_map)
            
            if tok.deprel not in feat_map.label_map:
                label_index = len(feat_map.label_map)
                feat_map.label_map[tok.deprel] = label_index
                feat_map.index_map[label_index] = tok.deprel
            
            distance = abs(tok.head - tok.id)
            if distance > max_distance:
                max_distance = distance
    
    # the maximum distance is stored in the otherwise unused deprel_map
    # and can be accessed with the MAX_DISTANCE string index
    feat_map.deprel_map[MAX_DISTANCE] = max_distance + 1

    return feat_map


def get_bilstm_gfeatures(sentence: Sentence, h: int, m: int) -> NNFeatures:
    """Returns an NNFeatures object with BiLSTM features for graph-based parsing.

    • The BiLSTM features are from Kiperwasser and Goldberg (2016).
    
    • Features (word, POS):
        • h[0], m[0]
    
    Args:
        sentence: A Sentence list of Tokens.
        h: An int token ID the head.
        m: An int token ID of the modifier.
    
    Returns:
        An NNFeatures object with BiLSTM features.
    """
    feats = NNFeatures()
    feats.words.append(sentence[h].form)
    feats.words.append(sentence[m].form)
    return feats


def get_feedforward_gfeatures(sentence: Sentence, h: int, m: int) -> NNFeatures:
    """Returns an NNFeatures object with feedforward features for graph-based parsing.

    • The 1-order-atomic feature set is from Pei, Ge, and Chang (2015) consisting of
    11 elements.
    
    • Features (word, POS, distance):
        • h[0], m[0], h[-1], m[-1], h[1], m[1], h[-2], m[-2], h[2], m[2], dis(h, m)
    
    Args:
        sentence: A Sentence list of Tokens.
        h: An int token ID the head.
        m: An int token ID of the modifier.
    
    Returns:
        An NNFeatures object with feedforward features.
    """
    feats = NNFeatures()
    # h[0], m[0]
    feats.words.append(sentence[h].form)
    feats.pos.append(sentence[h].upos)
    feats.words.append(sentence[m].form)
    feats.pos.append(sentence[m].upos)
    for i in range(1, 3):
        # h[-1], h[-2]
        if (h - i) >= 0:
            feats.words.append(sentence[h - i].form)
            feats.pos.append(sentence[h - i].upos)
        else:
            feats.words.append(NULL)
            feats.pos.append(NULL)
        # m[-1], m[-2]
        if (m - i) >= 0:
            feats.words.append(sentence[m - i].form)
            feats.pos.append(sentence[m - i].upos)
        else:
            feats.words.append(NULL)
            feats.pos.append(NULL)
        # h[1], h[2]
        if (h + i) < len(sentence):
            feats.words.append(sentence[h + i].form)
            feats.pos.append(sentence[h + i].upos)
        else:
            feats.words.append(NULL)
            feats.pos.append(NULL)
        # m[1], m[2]
        if (m + i) < len(sentence):
            feats.words.append(sentence[m + i].form)
            feats.pos.append(sentence[m + i].upos)
        else:
            feats.words.append(NULL)
            feats.pos.append(NULL)
    feats.distance = abs(h - m)
    return feats