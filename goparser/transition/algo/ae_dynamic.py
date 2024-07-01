"""This module contains the dynamic Arc-Eager oracle from Goldberg and Nivre (2012; 2013).
Builds a tree eagerly by adding arcs as soon as possible and 
attaches all left dependents first before right dependents.

The algorithm is equivalent to "nivreeager" in MaltParser.

By default, headless tokens are attached to the special ROOT node.
They can also be handled with the Unshift transition (Nivre and Fernández-González 2014).

Can only run projective-only parsing.

Supports Perceptron, Feedforward, and BiLSTM learning.
"""
from random import random

import numpy as np
import torch.optim as optim

from bilstm import BiLSTM
from feedforward import TFeedforward
from perceptron import BIAS_INDEX, AveragedMulticlassPerceptron, Matrix, MulticlassPerceptron
from p_features import PFeatureMap
from utils import Sentence

from transition.trans_feats import NNTFeatureExtractor
from transition.trans_utils import LEFT_ARC, RIGHT_ARC, SHIFT, REDUCE, \
    Arc, ArcSet, Configuration, OracleOutput, get_gold_tree

from transition.algo.arc_eager import ArcEager


class ArcEagerDynamic(ArcEager):
    """This class implements the Arc-Eager transition parser algorithm (Nivre 2003)
    with a dynamic oracle (Goldberg and Nivre 2012; 2013).
    Builds a tree eagerly by adding arcs as soon as possible and 
    attaches all left dependents first before right dependents.

    • The algorithm is equivalent to "nivreeager" in MaltParser.

    • By default, headless tokens are attached to the special ROOT node.
    They can also be handled with the unshift transition
    (Nivre and Fernández-González 2014).

    • Can only run projective-only parsing.

    • Supports Feedforward and BiLSTM learning.

    Attributes:
        unshift: An optional bool to enable the Unshift transition.
            Default: False
        root_label: An optional string to set the root label when headless tokens are
        attached to the special ROOT node.
            Default: None.
    """

    __slots__ = ('__k', '__p', '__agg_exp')
    
    def __init__(self,
                 num_loss_accum: int = 50,
                 unshift: bool = False,
                 root_label: str | None = None,
                 k: int = 2,
                 p: float = 0.1,
                 agg_exp: float = 1.0) -> None:
        """Inits a dynamic Arc-Eager transition algorithm.
        
        Args:
            num_loss_accum: The number accumulated losses necessary to run a backward pass.
                Default: 50.
            unshift: An optional bool to enable the Unshift transition.
                Default: False
            root_label: An optional string to set the root label when headless tokens are
            attached to the special ROOT node.
                Default: None.
            k: An int to set the first k iterations where the oracle transition is always chosen.
                Default: 2.
            p: A float to set the probability that the incorrect model transition is chosen.
                Default: 0.1.
            agg_exp: A float to set the margin constant where the wrong transition can be chosen
            even if it scores less than the correct transition but the difference is smaller than
            the margin.
                Default: 1.0.
        """
        super().__init__(False, num_loss_accum, unshift, root_label)
        self.__k = k
        self.__p = p
        self.__agg_exp = agg_exp
    
    def oracle(self,
               input_tokens: Sentence,
               features: PFeatureMap | NNTFeatureExtractor | None = None,
               model: MulticlassPerceptron | TFeedforward | BiLSTM | None = None,
               optimizer: optim.Optimizer | None = None) -> OracleOutput:
        sent_loss = 0.0
        num_trans = 0
        c = Configuration(input_tokens)
        gold_tree = get_gold_tree(input_tokens)
        
        # shift ROOT from buffer to stack
        c.stack.append(c.buffer.popleft())
        
        while c.buffer:
            la_cost = 0
            ra_cost = 0
            re_cost = 0
            sh_cost = 0
            
            s = c.stack[-1]
            b = c.buffer[0]
            c_arcs = ArcSet()
            for a in c.tree.values():
                c_arcs.update(a)
            
            # left-arc cost
            if len(c.stack) > 1:
                if not self._should_left_arc(c, gold_tree):
                    β = iter(c.buffer)
                    _ = next(β)
                    la_cost = len(
                        [k for k in β if (
                            Arc(k, s, c.input_tokens[s].deprel) in gold_tree[k]
                            or Arc(s, k, c.input_tokens[k].deprel) in gold_tree[s]
                        )]
                    )
            else:
                la_cost = 1
            
            # right-arc cost
            if not self._should_right_arc(c, gold_tree):
                β = iter(c.buffer)
                _ = next(β)
                ra_cost = len(
                    [k for k in c.stack[:-1] if (
                        Arc(k, b, c.input_tokens[b].deprel) in gold_tree[k]
                        or (
                            Arc(b, k, c.input_tokens[k].deprel) in gold_tree[b]
                            and len([a for a in c_arcs if a.dependent == k]) == 0
                        )
                    )]
                ) + len(
                    [k for k in β if Arc(k, b, c.input_tokens[b].deprel) in gold_tree[k]]
                )
            
            # reduce cost
            if len(c.stack) > 1:
                re_cost = len(
                    [k for k in c.buffer if Arc(s, k, c.input_tokens[k].deprel) in gold_tree[s]]
                )
            else:
                re_cost = 1
            
            # shift cost
            sh_cost = len(
                [k for k in c.stack if (
                    Arc(k, b, c.input_tokens[b].deprel) in gold_tree[k]
                    or (
                        Arc(b, k, c.input_tokens[k].deprel) in gold_tree[b]
                        and len([a for a in c_arcs if a.dependent == k]) == 0
                    )
                )]
            )
            
            costs = {
                LEFT_ARC + ' ' + c.input_tokens[c.stack[-1]].deprel: la_cost,
                RIGHT_ARC + ' ' + c.input_tokens[c.buffer[0]].deprel: ra_cost,
                REDUCE: re_cost,
                SHIFT: sh_cost
            }

            #################### PERCEPTRON ####################
            if isinstance(model, Matrix):
                x = features.get_feature_vector(c)
                φ_x = np.zeros(model.w.shape[1], dtype=np.int64)
                φ_x[x] = 1
                φ_x[BIAS_INDEX] = 1
                ŷ = np.dot(model.w, φ_x)
                
                # find highest scoring correct transition
                valid_trans = max(
                    [(score, i) for i, score in enumerate(ŷ) if (
                        features.index_map[i] in costs
                        and costs[features.index_map[i]] == 0
                    )],
                    default=(-1, -1) # no valid transition
                )
                
                # return if there are no more valid transitions when sentence is non-projective
                if valid_trans == (-1, -1):
                    # the sequence and instances list will be returned empty
                    return OracleOutput([], sent_loss, num_trans, [])

                # find highest scoring wrong transition
                wrong_trans = max(
                    (score, i) for i, score in enumerate(ŷ) if (
                        features.index_map[i] not in costs
                        or costs[features.index_map[i]] != 0
                    )
                )

                # update weights
                if valid_trans[0] < wrong_trans[0]:
                    model.w[valid_trans[1]] += φ_x
                    model.w[wrong_trans[1]] -= φ_x
                    if isinstance(model, AveragedMulticlassPerceptron):
                        model.u[valid_trans[1]] += np.dot(model.q, φ_x)
                        model.u[valid_trans[1]] -= np.dot(model.q, φ_x)
                    else:
                        model.correct += 1
                    model.total += 1
                
                # exploration
                best_valid = features.index_map[valid_trans[1]].split()[0]
                best_wrong = features.index_map[wrong_trans[1]].split()[0]
                next_trans = best_valid

                if self._i < self.__k:
                    pass
                elif not len(c.stack) > 1:
                    if best_wrong == LEFT_ARC or best_wrong == REDUCE:
                        pass
                elif valid_trans[0] > wrong_trans[0] and random() > self.__p:
                    pass
                else:
                    next_trans = best_wrong
                
                # updates
                self._do_transition(c, next_trans)
                num_trans += 1
                continue
            
            ################## NEURAL NETWORK ##################
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
            
            # return if there are no more valid transitions when sentence is non-projective
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
            best_valid = model.feature_map.index_map[valid_trans].split()[0]
            best_wrong = model.feature_map.index_map[wrong_trans].split()[0]
            next_trans = best_valid
            if self._i < self.__k:
                pass
            elif not len(c.stack) > 1:
                if best_wrong == LEFT_ARC or best_wrong == REDUCE:
                    pass
            elif output[0, valid_trans] - output[0, wrong_trans] > self.__agg_exp:
                pass
            elif output[0, valid_trans] > output[0, wrong_trans] and random() > self.__p:
                pass
            else:
                next_trans = best_wrong
            
            # updates
            self._do_transition(c, next_trans)
            num_trans += 1
        
        # the sequence and instances list will be returned empty
        return OracleOutput([], sent_loss, num_trans, [])