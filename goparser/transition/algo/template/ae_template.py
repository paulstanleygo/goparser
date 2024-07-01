"""This module contains the feature template for Arc-Eager from Zhang and Nivre (2011)."""
from utils import NULL

from transition.trans_utils import Configuration


def arc_eager_template(c: Configuration,
                       single_words: bool = True,
                       word_pairs: bool = True,
                       three_words: bool = True,
                       distance: bool = True,
                       valency: bool = True,
                       unigrams: bool = True,
                       third_order: bool = True,
                       label_set: bool = True) -> list[str]:
    """Returns a list of Arc-Eager features given a Configuration.
    
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
    """Adds baseline single word features for Arc-Eager given a Configuration and feature list.
    
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
    
    if len(c.buffer) > 2:
        N2 = c.input_tokens[c.buffer[2]]
        feats.append('N2wp' + N2.form + N2.upos)
        feats.append('N2w' + N2.form)
        feats.append('N2p' + N2.upos)
    else:
        feats.append('N2wp' + NULL)
        feats.append('N2w' + NULL)
        feats.append('N2p' + NULL)


def _word_pairs(c: Configuration, feats: list[str]) -> None:
    """Adds baseline word pair features for Arc-Eager given a Configuration and feature list.
    
    Args:
        c: A Configuration object.
        feats: A list of string features.
    """
    if c.stack and c.buffer:
        S0 = c.input_tokens[c.stack[-1]]
        N0 = c.input_tokens[c.buffer[0]]
        feats.append('S0wpN0wp' + S0.form + S0.upos + N0.form + N0.upos)
        feats.append('S0wpN0w' + S0.form + S0.upos + N0.form)
        feats.append('S0wN0wp' + S0.form + N0.form + N0.upos)
        feats.append('S0wpN0p' + S0.form + S0.upos + N0.upos)
        feats.append('S0pN0wp' + S0.upos + N0.form + N0.upos)
        feats.append('S0wN0w' + S0.form + N0.form)
        feats.append('S0pN0p' + S0.upos + N0.upos)
    else:
        feats.append('S0wpN0wp' + NULL)
        feats.append('S0wpN0w' + NULL)
        feats.append('S0wN0wp' + NULL)
        feats.append('S0wpN0p' + NULL)
        feats.append('S0pN0wp' + NULL)
        feats.append('S0wN0w' + NULL)
        feats.append('S0pN0p' + NULL)
    
    if len(c.buffer) > 1:
        feats.append('N0pN1p' + c.input_tokens[c.buffer[0]].upos
                     + c.input_tokens[c.buffer[1]].upos)
    else:
        feats.append('N0pN1p' + NULL)


def _three_words(c: Configuration, feats: list[str]) -> None:
    """Adds baseline three word features for Arc-Eager given a Configuration and feature list.
    
    Args:
        c: A Configuration object.
        feats: A list of string features.
    """
    if len(c.buffer) > 2:
        feats.append('N0pN1pN2p' + c.input_tokens[c.buffer[0]].upos
                     + c.input_tokens[c.buffer[1]].upos
                     + c.input_tokens[c.buffer[2]].upos)
    else:
        feats.append('N0pN1pN2p' + NULL)
    
    if c.stack and len(c.buffer) > 1:
        feats.append('S0pN0pN1p' + c.input_tokens[c.stack[-1]].upos
                     + c.input_tokens[c.buffer[0]].upos
                     + c.input_tokens[c.buffer[1]].upos)
    else:
        feats.append('S0pN0pN1p' + NULL)
    
    if c.stack and c.buffer and c.stack[-1] in c.has_head:
        feats.append('S0hpS0pN0p' + c.input_tokens[c.has_head[c.stack[-1]].head].upos
                        + c.input_tokens[c.stack[-1]].upos
                        + c.input_tokens[c.buffer[0]].upos)
    else:
        feats.append('S0hpS0pN0p' + NULL)
    
    if c.stack and c.buffer and c.stack[-1] in c.leftmost_child:
        S0l = c.leftmost_child[c.stack[-1]].dependent
        feats.append('S0pS0lpN0p' + c.input_tokens[c.stack[-1]].upos
                     + c.input_tokens[S0l].upos
                     + c.input_tokens[c.buffer[0]].upos)
    else:
        feats.append('S0pS0lpN0p' + NULL)
    
    if c.stack and c.buffer and c.stack[-1] in c.rightmost_child:
        S0r = c.rightmost_child[c.stack[-1]].dependent
        feats.append('S0pS0rpN0p' + c.input_tokens[c.stack[-1]].upos
                     + c.input_tokens[S0r].upos
                     + c.input_tokens[c.buffer[0]].upos)
    else:
        feats.append('S0pS0rpN0p' + NULL)
    
    if c.stack and c.buffer and c.buffer[0] in c.leftmost_child:
        N0l = c.leftmost_child[c.buffer[0]].dependent
        feats.append('S0pN0pN0lp' + c.input_tokens[c.stack[-1]].upos
                        + c.input_tokens[c.buffer[0]].upos
                        + c.input_tokens[N0l].upos)
    else:
        feats.append('S0pN0pN0lp' + NULL)


def _distance(c: Configuration, feats: list[str]) -> None:
    """Adds extended distance features for Arc-Eager given a Configuration and feature list.
    
    Args:
        c: A Configuration object.
        feats: A list of string features.
    """
    if c.stack and c.buffer:
        S0 = c.input_tokens[c.stack[-1]]
        N0 = c.input_tokens[c.buffer[0]]
        dis = str(c.buffer[0] - c.stack[-1])
        feats.append('S0wd' + S0.form + dis)
        feats.append('S0pd' + S0.upos + dis)
        feats.append('N0wd' + N0.form + dis)
        feats.append('N0pd' + N0.upos + dis)
        feats.append('S0wN0wd' + S0.form + N0.form + dis)
        feats.append('S0pN0pd' + S0.upos + N0.upos + dis)
    else:
        feats.append('S0wd' + NULL)
        feats.append('S0pd' + NULL)
        feats.append('N0wd' + NULL)
        feats.append('N0pd' + NULL)
        feats.append('S0wN0wd' + NULL)
        feats.append('S0pN0pd' + NULL)


def _valency(c: Configuration, feats: list[str]) -> None:
    """Adds extended distance features for Arc-Eager given a Configuration and feature list.
    
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
    
    if c.buffer and c.buffer[0] in c.tree:
        N0 = c.input_tokens[c.buffer[0]]
        vl = str(len(c.tree[c.buffer[0]]))
        feats.append('N0wvl' + N0.form + vl)
        feats.append('N0pvl' + N0.upos + vl)
    else:
        feats.append('N0wvl' + NULL)
        feats.append('N0pvl' + NULL)


def _unigrams(c: Configuration, feats: list[str]) -> None:
    """Adds extended unigram features for Arc-Eager given a Configuration and feature list.
    
    Args:
        c: A Configuration object.
        feats: A list of string features.
    """
    if c.stack and c.stack[-1] in c.has_head:
        S0h = c.input_tokens[c.has_head[c.stack[-1]].head]
        feats.append('S0hw' + S0h.form)
        feats.append('S0hp' + S0h.upos)
        feats.append('S0hl' + c.has_head[c.stack[-1]].deprel)
    else:
        feats.append('S0hw' + NULL)
        feats.append('S0hp' + NULL)
        feats.append('S0hl' + NULL)
    
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
    
    if c.buffer and c.buffer[0] in c.leftmost_child:
        N0l = c.leftmost_child[c.buffer[0]].dependent
        feats.append('N0lw' + c.input_tokens[N0l].form)
        feats.append('N0lp' + c.input_tokens[N0l].upos)
        feats.append('N0ll' + c.has_head[N0l].deprel)
    else:
        feats.append('N0lw' + NULL)
        feats.append('N0lp' + NULL)
        feats.append('N0ll' + NULL)


def _third_order(c: Configuration, feats: list[str]) -> None:
    """Adds extended third-order features for Arc-Eager given a Configuration and feature list.
    
    Args:
        c: A Configuration object.
        feats: A list of string features.
    """
    if (c.stack and c.stack[-1] in c.has_head
        and c.has_head[c.stack[-1]].head in c.has_head):
        S0h = c.has_head[c.stack[-1]]
        S0h2 = c.has_head[S0h.head]
        feats.append('S0h2w' + c.input_tokens[S0h2.head].form)
        feats.append('S0h2p' + c.input_tokens[S0h2.head].upos)
        feats.append('S0h2p' + S0h2.deprel)
        feats.append('S0pS0hpS0h2p' + c.input_tokens[c.stack[-1]].upos
                     + c.input_tokens[S0h.head].upos
                     + c.input_tokens[S0h2.head].upos)
    else:
        feats.append('S0h2w' + NULL)
        feats.append('S0h2p' + NULL)
        feats.append('S0h2p' + NULL)
        feats.append('S0pS0hpS0h2p' + NULL)
    
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
    
    if c.buffer and c.buffer[0] in c.second_leftmost_child:
        N0l2 = c.second_leftmost_child[c.buffer[0]].dependent
        feats.append('N0l2w' + c.input_tokens[N0l2].form)
        feats.append('N0l2p' + c.input_tokens[N0l2].upos)
        feats.append('N0l2l' + c.has_head[N0l2].deprel)
    else:
        feats.append('N0l2w' + NULL)
        feats.append('N0l2p' + NULL)
        feats.append('N0l2l' + NULL)
    
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
    
    if (c.buffer and c.buffer[0] in c.leftmost_child
        and c.buffer[0] in c.second_leftmost_child):
        N0 = c.input_tokens[c.buffer[0]]
        N0l = c.input_tokens[c.leftmost_child[c.buffer[0]].dependent]
        N0l2 = c.input_tokens[c.second_leftmost_child[c.buffer[0]].dependent]
        feats.append('N0pN0lpN0l2p' + N0.upos + N0l.upos + N0l2.upos)
    else:
        feats.append('N0pN0lpN0l2p' + NULL)
    
    return feats


def _label_set(c: Configuration, feats: list[str]) -> None:
    """Adds extended label set features for Arc-Eager given a Configuration and feature list.
    
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
    
    if c.buffer and c.buffer[0] in c.tree:
        N0 = c.input_tokens[c.buffer[0]]
        sl = sorted({a.deprel for a in c.tree[c.stack[-1]]})
        feats.append('N0wsl' + N0.form + ''.join(sl))
        feats.append('N0psl' + N0.upos + ''.join(sl))
    else:
        feats.append('N0wsl' + NULL)
        feats.append('N0psl' + NULL)