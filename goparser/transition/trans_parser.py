"""This module contains an implementation of a transition-based parser."""
from random import shuffle

import torch

from bilstm import BiLSTM
from dep_parser import NNDependencyParser, PDependencyParser
from p_features import PFeatures
from utils import DevelopmentSet, Treebank, stopwatch

from transition.trans_utils import set_arcs


class NNTransitionParser(NNDependencyParser):
    """This class implements a transition-based parser for a neural network."""
    
    def predict(self, treebank: Treebank, state_dict: str | None = None) -> None:
        """Predicts the dependency tree given a Treebank.
        
        Args:
            treebank: A Treebank list containing a treebank.
            state_dict: An optional string path to a saved PyTorch state_dict.
        """
        if state_dict is not None:
            print('Loading state_dict: ' + state_dict)
            self.model.load_state_dict(torch.load(state_dict, self.model.device))
        with torch.no_grad():
            self.model.eval()
            for sentence in treebank:
                if isinstance(self.model, BiLSTM):
                    self.model.init_state()
                    self.model.init_bilstm_vectors(sentence)
                
                set_arcs(sentence, self.algo.parse(sentence,
                                                   self.feature_extractor,
                                                   self.model))
    
    @stopwatch
    def train(self,
              train_treebank: Treebank,
              dev_set: DevelopmentSet,
              state_dict: str) -> None:
        """Trains a model on a treebank.
        
        Args:
            train_treebank: A Treebank list containing the training treebank.
            dev_set: A DevelopmentSet tuple containing the development sets.
            state_dict: A string path to save the state_dict or load a preexisting one.
        """
        optimizer = self.optimizer(self.model.parameters(), self.learning_rate)
        highest_dev_score = 0.0
        
        for e in range(self.epochs):
            print(f'Epoch {e + 1}')
            total_loss = 0.0
            total_trans = 0
            self.model.train()
            shuffle(train_treebank)
            
            for sentence in train_treebank:
                if isinstance(self.model, BiLSTM):
                    self.model.init_state()
                    self.model.init_bilstm_vectors(sentence)
                
                _, sent_loss, num_trans, _ = self.algo.oracle(sentence,
                                                              self.feature_extractor,
                                                              self.model,
                                                              optimizer)
                
                self.algo.accum_backward(self.model, optimizer)
                total_loss += sent_loss
                total_trans += num_trans
            
            self.algo.accum_backward(self.model, optimizer, force=True)
            self.algo.add_iteration()
            
            print('Training Set Loss: %.2f%%' % (total_loss / total_trans))
            highest_dev_score = self.validate(dev_set, highest_dev_score, state_dict)
            print('*FINISH*')


class PTransitionParser(PDependencyParser):
    """This class implements a transition-based parser for a perceptron."""
    
    def predict(self, treebank: Treebank):
        for sentence in treebank:
            set_arcs(sentence, self.algo.parse(sentence, self.feature_map, self.model))
    
    @stopwatch
    def train_instances(self,
                        instances: list[PFeatures],
                        dev_set: DevelopmentSet,
                        weights_file: str) -> None:
        """Trains a model on a list of PFeatures instances.
        
         Args:
            instances: A PFeatures list containing the training instances.
            dev_set: A DevelopmentSet tuple containing the development sets.
            weights_file: A string path to save the weights file.
        """
        highest_dev_score = 0
        for e in range(self.epochs):
            print(f'Epoch {e + 1}')
            shuffle(instances)
            for instance in instances:
                self.model.train(instance.feats, instance.label)
            print('Training Set Accuracy Score: %d / %d = %.2f%%' % (
                self.model.correct,
                self.model.total,
                self.model.get_score() * 100
            ))
            highest_dev_score = self.validate(dev_set, highest_dev_score, weights_file)
            print('*FINISH*')
    
    @stopwatch
    def train(self,
              train_treebank: Treebank,
              dev_set: DevelopmentSet,
              weights_file: str) -> None:
        """Trains a model on a treebank.
        
         Args:
            train_treebank: A Treebank list containing the training treebank.
            dev_set: A DevelopmentSet tuple containing the development sets.
            weights_file: A string path to save the weights file.
        """
        highest_dev_score = 0
        for e in range(self.epochs):
            print(f'Epoch {e + 1}')
            shuffle(train_treebank)
            for sentence in train_treebank:
                _, _, _, _ = self.algo.oracle(sentence,
                                              self.feature_map,
                                              self.model)
            print('Training Set Accuracy Score: %d / %d = %.2f%%' % (
                self.model.correct,
                self.model.total,
                self.model.get_score() * 100
            ))
            highest_dev_score = self.validate(dev_set, highest_dev_score, weights_file)
            print('*FINISH*')