"""This module contains functions for extracting
features for a transition-based neural parser.

The feedforward features are from Chen and Manning (2014).

The BiLSTM features are from Kiperwasser and Goldberg (2016).
Ideal minimal number of BiLSTM features are 3 for Arc-Standard and
2 for Arc-Eager and Arc-Hybrid (Shi, Huang, and Lee 2017).
"""
from collections.abc import Callable

from nn_features import NNFeatureMap, NNFeatures
from p_features import PFeatureMap, PFeatures
from utils import NULL, UNK, Sentence, Treebank

from transition.trans_utils import LEFT_ARC, RIGHT_ARC, SHIFT, REDUCE_SWAP, \
    Arc, Configuration, OracleOutput


NNTFeatureExtractor = Callable[[Configuration], NNFeatures]
"""Type alias for a transition-based neural network feature extractor function that returns
an NNFeatures object given a Configuration."""


def index_nntfeatures(treebank: Treebank,
                      oracle: Callable[[Sentence], OracleOutput]) -> NNFeatureMap:
    """Returns an NNFeatureMap object that maps transition features to index given a
    Treebank and oracle.
    
    Args:
        treebank: A Treebank list containing a treebank.
        oracle: A function that returns a list of transition sequences.
    
    Returns:
        An NNFeatureMap object with feature-to-index maps.
    """
    feat_map = NNFeatureMap()
    
    feat_map.word_map[NULL] = 0
    feat_map.pos_map[NULL] = 0
    feat_map.deprel_map[NULL] = 0

    feat_map.word_map[UNK] = 1
    feat_map.pos_map[UNK] = 1
    feat_map.deprel_map[UNK] = 1

    for sent in treebank:
        for tok in sent:
            if tok.form not in feat_map.word_map:
                word_index = len(feat_map.word_map)
                feat_map.word_map[tok.form] = word_index
                feat_map.wordfreq_map[word_index] = 1
            else:
                feat_map.wordfreq_map[feat_map.word_map[tok.form]] += 1
            
            if tok.upos not in feat_map.pos_map:
                feat_map.pos_map[tok.upos] = len(feat_map.pos_map)
            
            if tok.deprel not in feat_map.deprel_map:
                feat_map.deprel_map[tok.deprel] = len(feat_map.deprel_map)
            
        seq, _, _, _ = oracle(sent)
        for l in seq:
            if l not in feat_map.label_map:
                label_index = len(feat_map.label_map)
                feat_map.label_map[l] = label_index
                feat_map.index_map[label_index] = l

    return feat_map


def index_dynamic_nntfeatures(treebank: Treebank) -> NNFeatureMap:
    """Returns an NNFeatureMap object that maps dynamic oracle
    transition features to index given a Treebank.
    
    Args:
        treebank: A Treebank list containing a treebank.
    
    Returns:
        An NNFeatureMap object with feature-to-index maps.
    """
    feat_map = NNFeatureMap()
    
    feat_map.word_map[NULL] = 0
    feat_map.pos_map[NULL] = 0
    feat_map.deprel_map[NULL] = 0

    feat_map.word_map[UNK] = 1
    feat_map.pos_map[UNK] = 1
    feat_map.deprel_map[UNK] = 1

    feat_map.label_map[SHIFT] = 0
    feat_map.index_map[0] = SHIFT

    feat_map.label_map[REDUCE_SWAP] = 1
    feat_map.index_map[1] = REDUCE_SWAP

    for sent in treebank:
        for tok in sent:
            if tok.form not in feat_map.word_map:
                word_index = len(feat_map.word_map)
                feat_map.word_map[tok.form] = word_index
                feat_map.wordfreq_map[word_index] = 1
            else:
                feat_map.wordfreq_map[feat_map.word_map[tok.form]] += 1
            
            if tok.upos not in feat_map.pos_map:
                feat_map.pos_map[tok.upos] = len(feat_map.pos_map)
            
            if tok.deprel not in feat_map.deprel_map:
                feat_map.deprel_map[tok.deprel] = len(feat_map.deprel_map)
                label_index = len(feat_map.label_map)
                feat_map.label_map[LEFT_ARC + ' ' + tok.deprel] = label_index
                feat_map.index_map[label_index] = LEFT_ARC + ' ' + tok.deprel
                label_index = len(feat_map.label_map)
                feat_map.label_map[RIGHT_ARC + ' ' + tok.deprel] = label_index
                feat_map.index_map[label_index] = RIGHT_ARC + ' ' + tok.deprel

    return feat_map


def index_ptfeatures(treebank: Treebank,
                     oracle: Callable[[Sentence, PFeatureMap], OracleOutput],
                     template: Callable[[Configuration], list[str]]
) -> tuple[PFeatureMap, list[PFeatures]]:
    """Returns an PFeatureMap object that maps transition features to index given a
    Treebank and oracle.
    
    Args:
        treebank: A Treebank list containing a treebank.
        oracle: A function that returns a list of transition sequences and instances.
        template: A function that returns a list of feature strings given a Configuration.
    
    Returns:
        An PFeatureMap object with feature-to-index maps and a list of PFeatures instances.
    """
    feat_map = PFeatureMap()
    feat_map.template = template
    instances = list[PFeatures]()

    for sent in treebank:
        seq, _, _, ins = oracle(sent, feat_map)
        instances.extend(ins)
        for l in seq:
            if l not in feat_map.label_map:
                label_index = len(feat_map.label_map)
                feat_map.label_map[l] = label_index
                feat_map.index_map[label_index] = l
    
    feat_map.freeze = True

    return feat_map, instances


def index_dynamic_ptfeatures(treebank: Treebank,
                             oracle: Callable[[Sentence, PFeatureMap], OracleOutput],
                             template: Callable[[Configuration], list[str]]) -> PFeatureMap:
    """Returns an PFeatureMap object that maps dynamic oracle
    transition features to index given a Treebank.
    
    Args:
        treebank: A Treebank list containing a treebank.
        oracle: A function that returns a list of transition sequences and instances.
        template: A function that returns a list of feature strings given a Configuration.
    
    Returns:
        An PFeatureMap object with feature-to-index maps.
    """
    feat_map = PFeatureMap()
    feat_map.template = template
    
    feat_map.label_map[SHIFT] = 0
    feat_map.index_map[0] = SHIFT

    feat_map.label_map[REDUCE_SWAP] = 1
    feat_map.index_map[1] = REDUCE_SWAP

    for sent in treebank:
        _, _, _, _ = oracle(sent, feat_map)
        for tok in sent:
            la_label = LEFT_ARC + ' ' + tok.deprel
            if la_label not in feat_map.label_map:
                label_index = len(feat_map.label_map)
                feat_map.label_map[la_label] = label_index
                feat_map.index_map[label_index] = la_label
            ra_label = RIGHT_ARC + ' ' + tok.deprel
            if ra_label not in feat_map.label_map:
                label_index = len(feat_map.label_map)
                feat_map.label_map[ra_label] = label_index
                feat_map.index_map[label_index] = ra_label
    
    feat_map.freeze = True

    return feat_map


def get_bilstm_tfeatures(c: Configuration, num_features: int) -> NNFeatures:
    """Returns an NNFeatures object with BiLSTM features given a Configuration state.

    • A 2 element feature set is recommended for Arc-Eager and Arc-Hybrid (Shi et al. 2017).
    • Features (word, POS): S[0], B[0]

    • A 3 element feature set is recommended for Arc-Standard (Shi et al. 2017).
    • Features (word, POS): S[1], S[0], B[0]
    
    • A 4 element feature set is the original from Kiperwasser and Goldberg (2016).
    • Features (word, POS): S[2], S[1], S[0], B[0]

    Args:
        c: A Configuration object.
        num_features: An int 2, 3, or 4 for the number of features to be extracted.
    
    Returns:
        An NNFeatures object with BiLSTM features.
    """
    if num_features < 2 or num_features > 4:
        raise ValueError('the value of \'num_features\' must be 2, 3, or 4')
    feats = NNFeatures()
    while num_features > 1:
        num_features -= 1
        if len(c.stack) >= num_features:
            t = c.input_tokens[c.stack[-num_features]]
            feats.words.append(t.form)
        else:
            feats.words.append(NULL)
    if c.buffer:
        feats.words.append(c.input_tokens[c.buffer[0]].form)
    else:
        feats.words.append(NULL)
    return feats


def get_feedforward_tfeatures(c: Configuration,
                              b0_left_children: bool,
                              s0_head: bool) -> NNFeatures:
    """Returns an NNFeatures object with feedforward features given a Configuration state.

    • The feature set is from Chen and Manning (2014) consisting of 18 elements.
    
    • Features (word, POS, deprels):
        • S[0], lc1(S[0]), lc2(S[0]), lc1(lc1(S[0])), rc1(S[0]), rc2(S[0]), rc1(rc1(S[0])),
        
        • S[1], lc1(S[1]), lc2(S[1]), lc1(lc1(S[1])), rc1(S[1]), rc2(S[1]), rc1(rc1(S[1])),

        • S[2], B[0], B[1], B[2]
    
    • Setting b0_left_children to True will add lc1(B[0]), lc2(B[0]), lc1(lc1(B[0]))
    for Arc-Eager and Arc-Hybrid.

    • Setting s0_head to True will add h(S[0]) for Arc-Eager.
    
    Args:
        c: A Configuration object.
        b0_left_children: A bool for adding the children of the front buffer token.
        s0_head: A bool for adding the head of the top stack token.
    
    Returns:
        An NNFeatures object with feedforward features.
    """
    feats = NNFeatures()
    stack = (reversed(c.stack), len(c.stack), 'S')
    buffer = (iter(c.buffer), len(c.buffer), 'B')
    for it in stack, buffer:
        for i in range(3):
            if it[1] > i:
                t = c.input_tokens[next(it[0])]
                feats.words.append(t.form)
                feats.pos.append(t.upos)
                if it[2] == 'S' and i < 2:
                    _add_children(t.id, feats, c.input_tokens,
                                  c.leftmost_child, c.second_leftmost_child)
                    _add_children(t.id, feats, c.input_tokens,
                                  c.rightmost_child, c.second_rightmost_child)
            else:
                feats.words.append(NULL)
                feats.pos.append(NULL)
                if it[2] == 'S' and i < 2:
                    _no_children(feats)
                    _no_children(feats)
    if b0_left_children:
        if c.buffer:
            _add_children(c.buffer[0], feats, c.input_tokens,
                          c.leftmost_child, c.second_leftmost_child)
        else:
            _no_children(feats)
    if s0_head:
        if c.stack and c.stack[-1] in c.has_head:
            t = c.input_tokens[c.has_head[c.stack[-1]].head]
            feats.words.append(t.form)
            feats.pos.append(t.upos)
            feats.deprels.append(t.deprel)
        else:
            _null_feats(feats)
    return feats


def _add_children(head: int,
                  feats: NNFeatures,
                  input_tokens: Sentence,
                  furthest_child: dict[int, Arc],
                  second_furthest_child: dict[int, Arc]) -> None:
    """Adds the children to an NNFeatures object.
    
    Args:
        head: An int ID of the head.
        feats: An NNFeatures object.
        input_tokens: A Sentence container of input tokens.
        furthest_child: A dict containing the int ID of furthest child indexed by the head.
        second_furthest_child: A dict containing the int ID of second furthest child indexed
        by the head.
    """
    if head in furthest_child:
        c1 = input_tokens[furthest_child[head].dependent]
        feats.words.append(c1.form)
        feats.pos.append(c1.upos)
        feats.deprels.append(furthest_child[head].deprel)
        if head in second_furthest_child:
            c2 = input_tokens[second_furthest_child[head].dependent]
            feats.words.append(c2.form)
            feats.pos.append(c2.upos)
            feats.deprels.append(second_furthest_child[head].deprel)
        else:
            _null_feats(feats)
        if c1.id in furthest_child:
            c1c1 = input_tokens[furthest_child[c1.id].dependent]
            feats.words.append(c1c1.form)
            feats.pos.append(c1c1.upos)
            feats.deprels.append(furthest_child[c1.id].deprel)
        else:
            _null_feats(feats)
    else:
        _no_children(feats)


def _null_feats(feats: NNFeatures) -> None:
    """Adds null features to an NNFeatures object."""
    feats.words.append(NULL)
    feats.pos.append(NULL)
    feats.deprels.append(NULL)


def _no_children(feats: NNFeatures) -> None:
    """Adds no children features to an NNFeatures object."""
    _null_feats(feats)
    _null_feats(feats)
    _null_feats(feats)