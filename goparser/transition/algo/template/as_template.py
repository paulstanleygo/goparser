"""This module contains the feature template for Arc-Standard
adapted from Zhang and Nivre (2011)."""
from utils import NULL

from transition.trans_utils import Configuration


def arc_standard_template(c: Configuration,
                          single_words: bool = True,
                          word_pairs: bool = True,
                          three_words: bool = True,
                          distance: bool = True,
                          valency: bool = True,
                          unigrams: bool = True,
                          third_order: bool = True,
                          label_set: bool = True) -> list[str]:
    """Returns a list of Arc-Standard features given a Configuration.
    
    Args:
        c: A Configuration object.
        single_words: A bool to add single word features.
            Default: True.
        word_pairs: A bool to add word pair features.
            Default: True.
        three_words: A bool to add three word features.
            Default: True.
        distance: A bool to add distance features.
            Default: True.
        valency: A bool to add valency features.
            Default: True.
        unigrams: A bool to add unigram features.
            Default: True.
        third_order: A bool to add third-order features.
            Default: True.
        label_set: A bool to add label set features.
            Default: True.
    
    Returns:
        A list of feature strings.
    """
    feats = list[str]()
    if single_words:
        _single_words(c, feats)
    if word_pairs:
        _word_pairs(c, feats)
    if three_words:
        _three_words(c, feats)
    if distance:
        _distance(c, feats)
    if valency:
        _valency(c, feats)
    if unigrams:
        _unigrams(c, feats)
    if third_order:
        _third_order(c, feats)
    if label_set:
        _label_set(c, feats)
    return feats


def _single_words(c: Configuration, feats: list[str]) -> None:
    """Adds baseline single word features for Arc-Standard given a Configuration and feature list.
    
    Args:
        c: A Configuration object.
        feats: A list of string features.
    """
    if c.stack:
        S0 = c.input_tokens[c.stack[-1]]
        feats.append('S0wp' + S0.form + S0.upos)
        feats.append('S0w' + S0.form)
        feats.append('S0p' + S0.upos)
    else:
        feats.append('S0wp' + NULL)
        feats.append('S0w' + NULL)
        feats.append('S0p' + NULL)
    
    if len(c.stack) > 1:
        S1 = c.input_tokens[c.stack[-2]]
        feats.append('S1wp' + S1.form + S1.upos)
        feats.append('S1w' + S1.form)
        feats.append('S1p' + S1.upos)
    else:
        feats.append('S1wp' + NULL)
        feats.append('S1w' + NULL)
        feats.append('S1p' + NULL)
    
    if c.buffer:
        N0 = c.input_tokens[c.buffer[0]]
        feats.append('N0wp' + N0.form + N0.upos)
        feats.append('N0w' + N0.form)
        feats.append('S0p' + N0.upos)
    else:
        feats.append('N0wp' + NULL)
        feats.append('N0w' + NULL)
        feats.append('N0p' + NULL)
    
    if len(c.buffer) > 1:
        N1 = c.input_tokens[c.buffer[1]]
        feats.append('N1wp' + N1.form + N1.upos)
        feats.append('N1w' + N1.form)
        feats.append('N1p' + N1.upos)
    else:
        feats.append('N1wp' + NULL)
        feats.append('N1w' + NULL)
        feats.append('N1p' + NULL)


def _word_pairs(c: Configuration, feats: list[str]) -> None:
    """Adds baseline word pair features for Arc-Standard given a Configuration and feature list.
    
    Args:
        c: A Configuration object.
        feats: A list of string features.
    """
    if len(c.stack) > 1:
        S0 = c.input_tokens[c.stack[-1]]
        S1 = c.input_tokens[c.stack[-2]]
        feats.append('S0wpS1wp' + S0.form + S0.upos + S1.form + S1.upos)
        feats.append('S0wpS1w' + S0.form + S0.upos + S1.form)
        feats.append('S0wS1wp' + S0.form + S1.form + S1.upos)
        feats.append('S0wpS1p' + S0.form + S0.upos + S1.upos)
        feats.append('S0pS1wp' + S0.upos + S1.form + S1.upos)
        feats.append('S0wS1w' + S0.form + S1.form)
        feats.append('S0pS1p' + S0.upos + S1.upos)
    else:
        feats.append('S0wpS1wp' + NULL)
        feats.append('S0wpS1w' + NULL)
        feats.append('S0wS1wp' + NULL)
        feats.append('S0wpS1p' + NULL)
        feats.append('S0pS1wp' + NULL)
        feats.append('S0wS1w' + NULL)
        feats.append('S0pS1p' + NULL)
    
    if c.stack and c.buffer:
        feats.append('S0pN0p' + c.input_tokens[c.stack[-1]].upos
                     + c.input_tokens[c.buffer[0]].upos)
    else:
        feats.append('S0pN0p' + NULL)


def _three_words(c: Configuration, feats: list[str]) -> None:
    """Adds baseline three word features for Arc-Standard given a Configuration and feature list.
    
    Args:
        c: A Configuration object.
        feats: A list of string features.
    """
    if c.stack and len(c.buffer) > 1:
        feats.append('S0pN0pN1p' + c.input_tokens[c.stack[-1]].upos
                     + c.input_tokens[c.buffer[0]].upos
                     + c.input_tokens[c.buffer[1]].upos)
    else:
        feats.append('S0pN0pN1p' + NULL)
    
    if len(c.stack) > 1 and c.buffer:
        feats.append('S1pS0pN0p' + c.input_tokens[c.stack[-2]].upos
                     + c.input_tokens[c.stack[-1]].upos
                     + c.input_tokens[c.buffer[0]].upos)
    else:
        feats.append('S1pS0pN0p' + NULL)
    
    if len(c.stack) > 1 and c.stack[-1] in c.leftmost_child:
        S0l = c.leftmost_child[c.stack[-1]].dependent
        feats.append('S0pS0lpS1p' + c.input_tokens[c.stack[-1]].upos
                     + c.input_tokens[S0l].upos
                     + c.input_tokens[c.stack[-2]].upos)
    else:
        feats.append('S0pS0lpS1p' + NULL)
    
    if len(c.stack) > 1 and c.stack[-1] in c.rightmost_child:
        S0r = c.rightmost_child[c.stack[-1]].dependent
        feats.append('S0pS0rpS1p' + c.input_tokens[c.stack[-1]].upos
                     + c.input_tokens[S0r].upos
                     + c.input_tokens[c.stack[-2]].upos)
    else:
        feats.append('S0pS0rpS1p' + NULL)
    
    if len(c.stack) > 1 and c.stack[-2] in c.leftmost_child:
        S1l = c.leftmost_child[c.stack[-2]].dependent
        feats.append('S0pS1pS1lp' + c.input_tokens[c.stack[-1]].upos
                        + c.input_tokens[c.stack[-2]].upos
                        + c.input_tokens[S1l].upos)
    else:
        feats.append('S0pS1pS1lp' + NULL)
    
    if len(c.stack) > 1 and c.stack[-2] in c.rightmost_child:
        S1r = c.rightmost_child[c.stack[-2]].dependent
        feats.append('S0pS1pS1rp' + c.input_tokens[c.stack[-1]].upos
                        + c.input_tokens[c.stack[-2]].upos
                        + c.input_tokens[S1r].upos)
    else:
        feats.append('S0pS1pS1rp' + NULL)


def _distance(c: Configuration, feats: list[str]) -> None:
    """Adds extended distance features for Arc-Standard given a Configuration and feature list.
    
    Args:
        c: A Configuration object.
        feats: A list of string features.
    """
    if len(c.stack) > 1:
        S0 = c.input_tokens[c.stack[-1]]
        S1 = c.input_tokens[c.stack[-2]]
        dis = abs(str(c.stack[-1] - c.stack[-2]))
        feats.append('S0wd' + S0.form + dis)
        feats.append('S0pd' + S0.upos + dis)
        feats.append('S1wd' + S1.form + dis)
        feats.append('S1pd' + S1.upos + dis)
        feats.append('S0wS1wd' + S0.form + S1.form + dis)
        feats.append('S0pS1pd' + S0.upos + S1.upos + dis)
    else:
        feats.append('S0wd' + NULL)
        feats.append('S0pd' + NULL)
        feats.append('S1wd' + NULL)
        feats.append('S1pd' + NULL)
        feats.append('S0wS1wd' + NULL)
        feats.append('S0pS1pd' + NULL)


def _valency(c: Configuration, feats: list[str]) -> None:
    """Adds extended distance features for Arc-Standard given a Configuration and feature list.
    
    Args:
        c: A Configuration object.
        feats: A list of string features.
    """
    if c.stack and c.stack[-1] in c.tree:
        S0 = c.input_tokens[c.stack[-1]]
        vr = len([a for a in c.tree[c.stack[-1]] if a.dependent > c.stack[-1]])
        if vr:
            feats.append('S0wvr' + S0.form + str(vr))
            feats.append('S0pvr' + S0.upos + str(vr))
        else:
            feats.append('S0wvr' + NULL)
            feats.append('S0pvr' + NULL)
        
        vl = abs(len(c.tree[c.stack[-1]]) - vr)
        if vl:
            feats.append('S0wvl' + S0.form + str(vl))
            feats.append('S0pvl' + S0.upos + str(vl))
        else:
            feats.append('S0wvl' + NULL)
            feats.append('S0pvl' + NULL)
    else:
        feats.append('S0wvr' + NULL)
        feats.append('S0pvr' + NULL)
        feats.append('S0wvl' + NULL)
        feats.append('S0pvl' + NULL)
    
    if len(c.stack) > 1 and c.stack[-2] in c.tree:
        S1 = c.input_tokens[c.stack[-2]]
        vr = len([a for a in c.tree[c.stack[-2]] if a.dependent > c.stack[-2]])
        if vr:
            feats.append('S1wvr' + S1.form + str(vr))
            feats.append('S1pvr' + S1.upos + str(vr))
        else:
            feats.append('S1wvr' + NULL)
            feats.append('S1pvr' + NULL)
        
        vl = abs(len(c.tree[c.stack[-2]]) - vr)
        if vl:
            feats.append('S1wvl' + S1.form + str(vl))
            feats.append('S1pvl' + S1.upos + str(vl))
        else:
            feats.append('S1wvl' + NULL)
            feats.append('S1pvl' + NULL)
    else:
        feats.append('S1wvr' + NULL)
        feats.append('S1pvr' + NULL)
        feats.append('S1wvl' + NULL)
        feats.append('S1pvl' + NULL)


def _unigrams(c: Configuration, feats: list[str]) -> None:
    """Adds extended unigram features for Arc-Standard given a Configuration and feature list.
    
    Args:
        c: A Configuration object.
        feats: A list of string features.
    """
    if c.stack and c.stack[-1] in c.leftmost_child:
        S0l = c.leftmost_child[c.stack[-1]].dependent
        feats.append('S0lw' + c.input_tokens[S0l].form)
        feats.append('S0lp' + c.input_tokens[S0l].upos)
        feats.append('S0ll' + c.has_head[S0l].deprel)
    else:
        feats.append('S0lw' + NULL)
        feats.append('S0lp' + NULL)
        feats.append('S0ll' + NULL)
    
    if c.stack and c.stack[-1] in c.rightmost_child:
        S0r = c.rightmost_child[c.stack[-1]].dependent
        feats.append('S0rw' + c.input_tokens[S0r].form)
        feats.append('S0rp' + c.input_tokens[S0r].upos)
        feats.append('S0rl' + c.has_head[S0r].deprel)
    else:
        feats.append('S0rw' + NULL)
        feats.append('S0rp' + NULL)
        feats.append('S0rl' + NULL)
    
    if len(c.stack) > 1 and c.stack[-2] in c.leftmost_child:
        S1l = c.leftmost_child[c.stack[-2]].dependent
        feats.append('S1lw' + c.input_tokens[S1l].form)
        feats.append('S1lp' + c.input_tokens[S1l].upos)
        feats.append('S1ll' + c.has_head[S1l].deprel)
    else:
        feats.append('S1lw' + NULL)
        feats.append('S1lp' + NULL)
        feats.append('S1ll' + NULL)
    
    if len(c.stack) > 1 and c.stack[-2] in c.rightmost_child:
        S1r = c.rightmost_child[c.stack[-2]].dependent
        feats.append('S1rw' + c.input_tokens[S1r].form)
        feats.append('S1rp' + c.input_tokens[S1r].upos)
        feats.append('S1rl' + c.has_head[S1r].deprel)
    else:
        feats.append('S1rw' + NULL)
        feats.append('S1rp' + NULL)
        feats.append('S1rl' + NULL)


def _third_order(c: Configuration, feats: list[str]) -> None:
    """Adds extended third-order features for Arc-Standard given a Configuration and feature list.
    
    Args:
        c: A Configuration object.
        feats: A list of string features.
    """
    if c.stack and c.stack[-1] in c.second_leftmost_child:
        S0l2 = c.second_leftmost_child[c.stack[-1]].dependent
        feats.append('S0l2w' + c.input_tokens[S0l2].form)
        feats.append('S0l2p' + c.input_tokens[S0l2].upos)
        feats.append('S0l2l' + c.has_head[S0l2].deprel)
    else:
        feats.append('S0l2w' + NULL)
        feats.append('S0l2p' + NULL)
        feats.append('S0l2l' + NULL)
    
    if c.stack and c.stack[-1] in c.second_rightmost_child:
        S0l2 = c.second_rightmost_child[c.stack[-1]].dependent
        feats.append('S0r2w' + c.input_tokens[S0l2].form)
        feats.append('S0r2p' + c.input_tokens[S0l2].upos)
        feats.append('S0r2l' + c.has_head[S0l2].deprel)
    else:
        feats.append('S0r2w' + NULL)
        feats.append('S0r2p' + NULL)
        feats.append('S0r2l' + NULL)
    
    if len(c.stack) > 1 and c.stack[-2] in c.second_leftmost_child:
        S1l2 = c.second_leftmost_child[c.stack[-2]].dependent
        feats.append('S1l2w' + c.input_tokens[S1l2].form)
        feats.append('S1l2p' + c.input_tokens[S1l2].upos)
        feats.append('S1l2l' + c.has_head[S1l2].deprel)
    else:
        feats.append('S1l2w' + NULL)
        feats.append('S1l2p' + NULL)
        feats.append('S1l2l' + NULL)
    
    if len(c.stack) > 1 and c.stack[-2] in c.second_rightmost_child:
        S1l2 = c.second_rightmost_child[c.stack[-2]].dependent
        feats.append('S1r2w' + c.input_tokens[S1l2].form)
        feats.append('S1r2p' + c.input_tokens[S1l2].upos)
        feats.append('S1r2l' + c.has_head[S1l2].deprel)
    else:
        feats.append('S1r2w' + NULL)
        feats.append('S1r2p' + NULL)
        feats.append('S1r2l' + NULL)
    
    if (c.stack and c.stack[-1] in c.leftmost_child
        and c.stack[-1] in c.second_leftmost_child):
        S0 = c.input_tokens[c.stack[-1]]
        S0l = c.input_tokens[c.leftmost_child[c.stack[-1]].dependent]
        S0l2 = c.input_tokens[c.second_leftmost_child[c.stack[-1]].dependent]
        feats.append('S0pS0lpS0l2p' + S0.upos + S0l.upos + S0l2.upos)
    else:
        feats.append('S0pS0lpS0l2p' + NULL)
    
    if (c.stack and c.stack[-1] in c.rightmost_child
        and c.stack[-1] in c.second_rightmost_child):
        S0 = c.input_tokens[c.stack[-1]]
        S0r = c.input_tokens[c.rightmost_child[c.stack[-1]].dependent]
        S0r2 = c.input_tokens[c.second_rightmost_child[c.stack[-1]].dependent]
        feats.append('S0pS0rpS0r2p' + S0.upos + S0r.upos + S0r2.upos)
    else:
        feats.append('S0pS0rpS0r2p' + NULL)
    
    if (len(c.stack) > 1 and c.stack[-2] in c.leftmost_child
        and c.stack[-2] in c.second_leftmost_child):
        S1 = c.input_tokens[c.stack[-2]]
        S1l = c.input_tokens[c.leftmost_child[c.stack[-2]].dependent]
        S1l2 = c.input_tokens[c.second_leftmost_child[c.stack[-2]].dependent]
        feats.append('S1pS1lpS1l2p' + S1.upos + S1l.upos + S1l2.upos)
    else:
        feats.append('S1pS1lpS1l2p' + NULL)
    
    if (len(c.stack) > 1 and c.stack[-2] in c.rightmost_child
        and c.stack[-2] in c.second_rightmost_child):
        S1 = c.input_tokens[c.stack[-2]]
        S1r = c.input_tokens[c.rightmost_child[c.stack[-2]].dependent]
        S1r2 = c.input_tokens[c.second_rightmost_child[c.stack[-2]].dependent]
        feats.append('S1pS1rpS1r2p' + S1.upos + S1r.upos + S1r2.upos)
    else:
        feats.append('S1pS1rpS1r2p' + NULL)


def _label_set(c: Configuration, feats: list[str]) -> None:
    """Adds extended label set features for Arc-Standard given a Configuration and feature list.
    
    Args:
        c: A Configuration object.
        feats: A list of string features.
    """
    if c.stack and c.stack[-1] in c.tree:
        S0 = c.input_tokens[c.stack[-1]]
        sr = sorted({a.deprel for a in c.tree[c.stack[-1]] if a.dependent > c.stack[-1]})
        if sr:
            feats.append('S0wsr' + S0.form + ''.join(sr))
            feats.append('S0psr' + S0.upos + ''.join(sr))
        else:
            feats.append('S0wsr' + NULL)
            feats.append('S0psr' + NULL)
        
        sl = sorted({a.deprel for a in c.tree[c.stack[-1]] if a.dependent < c.stack[-1]})
        if sl:
            feats.append('S0wsl' + S0.form + ''.join(sl))
            feats.append('S0psl' + S0.upos + ''.join(sl))
        else:
            feats.append('S0wsl' + NULL)
            feats.append('S0psl' + NULL)
    else:
        feats.append('S0wsr' + NULL)
        feats.append('S0psr' + NULL)
        feats.append('S0wsl' + NULL)
        feats.append('S0psl' + NULL)
    
    if len(c.stack) > 1 and c.stack[-1] in c.tree:
        S1 = c.input_tokens[c.stack[-2]]
        sr = sorted({a.deprel for a in c.tree[c.stack[-1]] if a.dependent > c.stack[-1]})
        if sr:
            feats.append('S1wsr' + S1.form + ''.join(sr))
            feats.append('S1psr' + S1.upos + ''.join(sr))
        else:
            feats.append('S1wsr' + NULL)
            feats.append('S1psr' + NULL)
        
        sl = sorted({a.deprel for a in c.tree[c.stack[-1]] if a.dependent < c.stack[-1]})
        if sl:
            feats.append('S1wsl' + S1.form + ''.join(sl))
            feats.append('S1psl' + S1.upos + ''.join(sl))
        else:
            feats.append('S1wsl' + NULL)
            feats.append('S1psl' + NULL)
    else:
        feats.append('S1wsr' + NULL)
        feats.append('S1psr' + NULL)
        feats.append('S1wsl' + NULL)
        feats.append('S1psl' + NULL)