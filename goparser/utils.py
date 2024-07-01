"""This module contains utility classes, functions, constants, and aliases
for dependency parsing."""
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
from time import perf_counter
from typing import Any, NamedTuple


ROOT = '_ROOT_'
"""String constant representing the special root token."""
NULL = '_NULL_'
"""String constant representing the special null token."""
UNK = '_UNK_'
"""String constant representing the special unknown token."""


@dataclass(slots=True)
class Token:
    """A class for storing token information relevant for dependency parsing.
    The attributes of this class correspond to the fields of the CoNLL-U format.
    The properties of this class ensure compatibility with the CoNLL-X format
    and are aliases to the corresponding CoNLL-U fields.
    
    When an instance of this class is converted to string,
    it will be formatted into CoNLL-U/CoNLL-X format.

    Attributes:
        id: An integer containing the word index.
        form: A string containing the word form or punctuation symbol.
        lemma: An optional string containing the lemma or stem of word form.
        upos: An optional string containing the universal part-of-speech tag.
        xpos: An optional string containing the language-specific part-of-speech tag.
        feats: An optional string containing the morphology features.
        head: An optional integer containing the ID number of the HEAD of the current word.
        deprel: An optional string containing the universal dependency relation to the HEAD.
        deps: An optional string containing the enhanced dependency graph in the form of 
        a list of head-deprel pairs.
        misc: An optional string containing any other annotation.
    
    Properties:
        cpostag: An optional string containing the coarse part-of-speech tag. 
        Alias of the 'upos' field.
        postag: An optional string containing the fine part-of-speech tag. 
        Alias of the 'xpos' field.
        phead: An optional integer containing the ID projective head of the current token. 
        Alias of the 'deps' field.
        pdeprel: An optional string containing the dependency relation to the PHEAD. 
        Alias of the 'misc' field.
    """

    id: int
    form: str
    lemma: str = None
    upos: str = None
    xpos: str = None
    feats: str = None
    head: int = None
    deprel: str = None
    deps: str = None
    misc: str = None

    @property
    def cpostag(self) -> str:
        return self.upos
    
    @cpostag.setter
    def cpostag(self, value: str) -> None:
        self.upos = value

    @property
    def postag(self) -> str:
        return self.xpos
    
    @postag.setter
    def postag(self, value: str) -> None:
        self.xpos = value

    @property
    def phead(self) -> int:
        return None if self.deps is None else int(self.deps)
    
    @phead.setter
    def phead(self, value: int) -> None:
        self.deps = str(value)

    @property
    def pdeprel(self) -> str:
        return self.misc
    
    @pdeprel.setter
    def pdeprel(self, value: str) -> None:
        self.misc = value

    def __str__(self) -> str:
        """Creates a string formatted in CoNLL-U/CoNLL-X format.
        
        Returns:
            A string in CoNLL-U/CoNLL-X format.
        """
        conll = str(self.id) + '\t' + self.form + '\t'
        conll += '_\t' if self.lemma == None else self.lemma + '\t'
        conll += '_\t' if self.upos == None else self.upos + '\t'
        conll += '_\t' if self.xpos == None else self.xpos + '\t'
        conll += '_\t' if self.feats == None else self.feats + '\t'
        conll += '_\t' if self.head == None else str(self.head) + '\t'
        conll += '_\t' if self.deprel == None else self.deprel + '\t'
        conll += '_\t' if self.deps == None else self.deps + '\t'
        conll += '_\n' if self.misc == None else self.misc + '\n'
        return conll

    @classmethod
    def conll(cls, line: str):
        """A class method which accepts a string in CoNLL-U or CoNLL-X format.

        Args:
            line: A string containing a token formatted in CoNLL-U or CoNLL-X.

        Returns:
            A new Token object.
        """
        line = line.split('\t')
        return cls(
            int(line[0]),                               # id
            line[1],                                    # form
            None if line[2] == '_' else line[2],        # lemma
            None if line[3] == '_' else line[3],        # upos / cpostag
            None if line[4] == '_' else line[4],        # xpos / postag
            None if line[5] == '_' else line[5],        # feats
            None if line[6] == '_' else int(line[6]),   # head
            None if line[7] == '_' else line[7],        # deprel
            None if line[8] == '_' else line[8],        # deps / phead
            None if line[9] == '_\n' else line[9]       # misc / pdeprel
        )
    
    @classmethod
    def conll09(cls, line: str):
        """A class method which accepts a string in CoNLL-2009 format.

        Args:
            line: A string containing a token formatted in CoNLL-2009.

        Returns:
            A new Token object.
        """
        line = line.split('\t')
        # get token ID for TIGER
        tok_id = line[0].split('_')
        line[0] = tok_id[0] if len(tok_id) == 1 else tok_id[1]
        return cls(
            int(line[0]),                               # id
            line[1],                                    # form
            None if line[2] == '_' else line[2],        # lemma
            None if line[4] == '_' else line[4],        # upos / cpostag
            None,                                       # xpos / postag
            None if line[6] == '_' else line[6],        # feats
            None if line[8] == '_' else int(line[8]),   # head
            None if line[10] == '_' else line[10],      # deprel
            None,                                       # deps / phead
            None                                        # misc / pdeprel
        )


ROOT_TOKEN = Token(
    id=0,
    form=ROOT,
    lemma=ROOT,
    upos=ROOT,
    xpos=ROOT,
    feats=None,
    head=-1,
    deprel=ROOT,
    deps=None,
    misc=None
)
"""Token constant representing the special root token."""


Sentence = list[Token]
"""Type alias for a list of Tokens."""


Treebank = list[Sentence]
"""Type alias for a list containing Sentence lists (lists of Tokens)."""


def read(treebank_file: str, conll09: bool = False) -> Treebank:
    """Reads a treebank formatted in CoNLL-U or CoNLL-X.
    
    To read a treebank in CoNLL-2009 format, set the conll09 argument to True.

    Args:
        treebank_file: A string containing the file name and path.
        conll09: A bool to specify that the input is in CoNLL-2009 format.
            Defaut: False.
    
    Returns:
        A Treebank list containing the treebank.
    """
    print('Reading treebank \'%s\'...' % (treebank_file))
    treebank = []
    sentence = [ROOT_TOKEN]
    init = True
    with open(treebank_file, 'r') as f:
        for line in f:
            if line != '\n' and not line.startswith('#'):
                token = Token.conll09(line) if conll09 else Token.conll(line)
                if int(token.id) == 1:
                    if init == True:
                        init = False
                    else:
                        treebank.append(sentence)
                        sentence = [ROOT_TOKEN]
                sentence.append(token)
        treebank.append(sentence)
    print('*FINISH*')
    return treebank


def write(treebank: Treebank, treebank_file: str) -> None:
    """Writes a treebank in a Treebank list to file.

    The file will be written in CoNLL-U/CoNLL-X format.

    Args:
        treebank: A Treebank list containing a treebank.
        treebank_file: A string containing the file name and path.
    """
    with open(treebank_file, 'w') as f:
        print('Writing treebank \'%s\'...' % (treebank_file))
        for sentence in treebank:
            for i in range(1, len(sentence)):
                f.write(str(sentence[i]))
            f.write('\n')
        print('*FINISH*')


class AttachmentScore(NamedTuple):
    """A namedtuple for storing the unlabeled and labeled attachment score.
    
    Fields:
        uas: A float containing the unlabeled attachment score.
        las: A float containing the labeled attachment score.
    """
    
    uas: float
    las: float


def get_attachment_scores(predicted: Treebank, gold: Treebank) -> AttachmentScore:
    """Returns the unlabeled and labelled attachment score.

    Args:
        predicted: A Treebank list with the predicted dependency arcs.
        gold: A Treebank list with the gold dependency arcs.
    
    Returns:
        An AttachmentScore namedtuple (uas, las) containing floats of 
        the unlabeled and labeled attachment score.
    """
    u_correct = 0
    l_correct = 0
    total = 0
    for i in range(len(gold)):
        for j in range(1, len(gold[i])):
            total += 1
            if predicted[i][j].head == gold[i][j].head:
                u_correct += 1
                if predicted[i][j].deprel == gold[i][j].deprel:
                    l_correct += 1
    uas = u_correct / total
    las = l_correct / total
    print('UAS: %d / %d = %.2f%%' % (
        u_correct,
        total,
        uas * 100
    ))
    print('LAS: %d / %d = %.2f%%' % (
        l_correct,
        total,
        las * 100
    ))
    return AttachmentScore(uas, las)


class DevelopmentSet(NamedTuple):
    """A namedtuple for storing the blind and gold development sets.
    
    Fields:
        blind: A Treebank containing the blind development set.
        gold: A Treebank containing the gold development set.
    """
    
    blind: Treebank
    gold: Treebank


def get_development_set(dev_treebank: Treebank) -> DevelopmentSet:
    """Returns a DevelopmentSet tuple with blind and gold development sets
    given a development Treebank.
    
    Args:
        dev_treebank: A Treebank list with the development Treebank.
    
    Returns:
        A DevelopmentSet namedtuple (blind, gold) containing Treebanks for
        the blind and gold development sets.
    """
    dev_blind = deepcopy(dev_treebank)
    for sentence in dev_blind:
        for token in sentence:
            token.head = None
            token.deprel = None
    return DevelopmentSet(dev_blind, dev_treebank)


def stopwatch(func: Callable) -> Any:
    """A function decorator that prints the elapsed time."""
    @wraps(func)
    def counter(*args, **kwargs) -> Any:
        start = perf_counter()
        result = func(*args, **kwargs)
        stop = perf_counter()
        print(f'Elapsed time: {str(timedelta(seconds=round(stop - start)))}')
        return result
    return counter