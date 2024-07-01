"""This module contains the static-dynamic Arc-Hybrid oracle from
Lhoneux, Stymne, and Nivre (2017).

Non-projective parsing is supported with the eager SWAP operation by default.

Projective-only parsing is also available by disabling the SWAP transition.

Supports Feedforward and BiLSTM learning.
"""
from collections import defaultdict
from random import random

import torch.optim as optim

from bilstm import BiLSTM
from feedforward import TFeedforward
from utils import Sentence

from transition.trans_feats import NNTFeatureExtractor
from transition.trans_utils import LEFT_ARC, RIGHT_ARC, SHIFT, SWAP, \
    Configuration, NonProjective, OracleOutput, \
    get_gold_tree, get_proj_dict, get_proj_order

from transition.algo.arc_hybrid import ArcHybrid


class ArcHybridStaticDynamic(ArcHybrid):
    """This class implements the Arc-Hybrid transition parser algorithm
    (Kuhlmann et al. 2011) with a static-dynamic oracle (Lhoneux, Stymne, and Nivre 2017).
    Builds a tree from the bottom up and attaches all left dependents first before right
    dependents.

    • Non-projective parsing is supported with the eager SWAP operation by default.

    • Projective-only parsing is also available by disabling the SWAP transition.

    • Supports Feedforward and BiLSTM learning.

    Attributes:
        non_proj: A NonProjective enum to set the non-projective parsing setting.
        Only NONE and EAGER are valid.
            Default: EAGER.
    """
    
    __slots__ = ('__k', '__p', '__agg_exp')

    def __init__(self,
                 num_loss_accum: int = 50,
                 non_proj: NonProjective = NonProjective.EAGER,
                 k: int = 2,
                 p: float = 0.1,
                 agg_exp: float = 1.0) -> None:
        """Inits a static-dynamic Arc-Hybrid transition algorithm.
        
        Args:
            num_loss_accum: The number accumulated losses necessary to run a backward pass.
                Default: 50.
            non_proj: A NonProjective enum to set the non-projective parsing setting.
            Only NONE and EAGER are valid.
                Default: EAGER.
            k: An int to set the first k iterations where the oracle transition is always chosen.
                Default: 2.
            p: A float to set the probability that the incorrect model transition is chosen.
                Default: 0.1.
            agg_exp: A float to set the margin constant where the wrong transition can be chosen
            even if it scores less than the correct transition but the difference is smaller than
            the margin.
                Default: 1.0.
        """
        if non_proj == NonProjective.LAZY:
            raise ValueError('non_proj must be NONE or EAGER.')
        super().__init__(False, num_loss_accum, non_proj)
        self.__k = k
        self.__p = p
        self.__agg_exp = agg_exp
    
    def oracle(self,
               input_tokens: Sentence,
               features: NNTFeatureExtractor,
               model: TFeedforward | BiLSTM,
               optimizer: optim.Optimizer) -> OracleOutput:
        sent_loss = 0.0
        num_trans = 0
        c = Configuration(input_tokens)
        gold_tree = get_gold_tree(input_tokens)
    
        # shift ROOT from buffer to stack
        c.stack.append(c.buffer.popleft())
        # shift next token to the stack
        c.stack.append(c.buffer.popleft())

        # the set of reachable dependents
        rdeps = defaultdict[int, set[int]](set[int])
        for token in input_tokens:
            rdeps[token.head].add(token.id)
        
        # check if tree is in projective order and activate swap if not
        proj_dict = None
        if self.non_proj != NonProjective.NONE:
            proj_list = get_proj_order(gold_tree)
            if proj_list != sorted(proj_list):
                proj_dict = get_proj_dict(proj_list)
    
        while not self._terminal(c):
            # force static swap
            if self._should_swap(c, gold_tree, proj_dict):
                la_cost = 1
                ra_cost = 1
                sh_cost = 1
                sw_cost = 0
            else:
                la_cost = 0
                ra_cost = 0
                sh_cost = 0
                sw_cost = 1
                
                # left-arc cost
                if self._can_left_arc(c):
                    la_cost = len(rdeps[c.stack[-1]])
                    if c.buffer[0] != c.input_tokens[c.stack[-1]].head:
                        if c.stack[-1] in rdeps[c.input_tokens[c.stack[-1]].head]:
                            la_cost += 1
                else:
                    la_cost = 1
                
                # right-arc cost
                if self._can_right_arc(c):
                    ra_cost = len(rdeps[c.stack[-1]])
                    if c.stack[-2] != c.input_tokens[c.stack[-1]].head:
                        if c.stack[-1] in rdeps[c.input_tokens[c.stack[-1]].head]:
                            ra_cost += 1
                else:
                    ra_cost = 1
                
                # shift cost
                if self._can_shift(c):
                    no_updates = False
                    if proj_dict:
                        for token in c.buffer:
                            if c.buffer[0] < token:
                                if proj_dict[c.buffer[0]] > proj_dict[token]:
                                    no_updates = True
                                    break
                    if not no_updates:
                        b0_head_in_stack = False
                        d_stack = {d for d in rdeps[c.buffer[0]] if d in c.stack}
                        sh_cost = len(d_stack)
                        if c.buffer[0] in rdeps[c.input_tokens[c.buffer[0]].head]:
                            if c.input_tokens[c.buffer[0]].head in c.stack[:-1]:
                                sh_cost += 1
                                b0_head_in_stack = True
                else:
                    sh_cost = 1
            
            costs = {
                LEFT_ARC + ' ' + c.input_tokens[c.stack[-1]].deprel: la_cost,
                RIGHT_ARC + ' ' + c.input_tokens[c.stack[-1]].deprel: ra_cost,
                SHIFT: sh_cost,
                SWAP: sw_cost
            }
            
            # model prediction
            if not self._loss_accum:
                optimizer.zero_grad()
            output = model(features(c))
            
            # find highest scoring correct transition
            valid_trans = max(
                [(score, i) for i, score in enumerate(output[0]) if (
                    model.feature_map.index_map[i] in costs
                    and costs[model.feature_map.index_map[i]] == 0
                )],
                default=(-1, -1) # no valid transition
            )[1]
            
            # return if there are no more valid transitions when in projective only mode
            if valid_trans == -1:
                # the sequence and instances list will be returned empty
                return OracleOutput([], sent_loss, num_trans, [])
            
            # find highest scoring wrong transition
            wrong_trans = max(
                (score, i) for i, score in enumerate(output[0]) if (
                    model.feature_map.index_map[i] not in costs
                    or costs[model.feature_map.index_map[i]] != 0
                )
            )[1]

            # transition hinge loss
            if output[0, valid_trans] < output[0, wrong_trans] + 1.0:
                loss = output[0, wrong_trans] - output[0, valid_trans]
                self._loss_accum.append(loss)
                sent_loss += loss.item()
                
            # aggressive exploration
            if costs[SWAP] == 0:
                next_trans = SWAP
            else:
                best_valid = model.feature_map.index_map[valid_trans].split()[0]
                best_wrong = model.feature_map.index_map[wrong_trans].split()[0]
                next_trans = best_valid
                if self._i < self.__k:
                    pass
                elif best_wrong == SWAP:
                    pass
                elif best_wrong == LEFT_ARC and not self._can_left_arc(c):
                    pass
                elif best_wrong == RIGHT_ARC and not self._can_right_arc(c):
                    pass
                elif best_wrong == SHIFT and not self._can_shift(c):
                    pass
                elif output[0, valid_trans] - output[0, wrong_trans] > self.__agg_exp:
                    pass
                elif output[0, valid_trans] > output[0, wrong_trans] and random() > self.__p:
                    pass
                else:
                    next_trans = best_wrong
            
            # updates
            if next_trans == LEFT_ARC:
                rdeps[c.stack[-1]] = set[int]()
                rdeps[c.input_tokens[c.stack[-1]].head].discard(c.stack[-1])
            elif next_trans == RIGHT_ARC:
                rdeps[c.stack[-1]] = set[int]()
                rdeps[c.input_tokens[c.stack[-1]].head].discard(c.stack[-1])
            elif next_trans == SHIFT:
                if not no_updates:
                    rdeps[c.buffer[0]] -= d_stack
                    if b0_head_in_stack:
                        rdeps[c.input_tokens[c.buffer[0]].head].discard(c.buffer[0])
            self._do_transition(c, next_trans)
            num_trans += 1
        
        # the sequence and instances list will be returned empty
        return OracleOutput([], sent_loss, num_trans, [])