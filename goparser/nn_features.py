"""This module contains dataclasses for storing and mapping
features and class labels for a neural network.
"""
from dataclasses import dataclass, field
import json
import os
import shutil
import tarfile


@dataclass(eq=False, frozen=True, slots=True)
class NNFeatureMap:
    """This dataclass stores the feature-to-index maps.
    
    Attributes:
        label_map: A dictionary mapping label strings to index.
        index_map: A dictionary mapping index to label strings.
        word_map: A dictionary mapping word strings to index.
        wordfreq_map: A dictionary mapping word index to integer frequency.
        pos_map: A dictionary mapping part-of-speech strings to index.
        deprel_map: A dictionary mapping dependency relation strings to index.
    """

    label_map: dict[str, int] = field(default_factory=dict)
    index_map: dict[int, str] = field(default_factory=dict)
    word_map: dict[str, int] = field(default_factory=dict)
    wordfreq_map: dict[int, int] = field(default_factory=dict)
    pos_map: dict[str, int] = field(default_factory=dict)
    deprel_map: dict[str, int] = field(default_factory=dict)
    
    @classmethod
    def open(cls, features_file: str):
        """Opens an NNFeatureMap file.

        Args:
            features_file: A string containing the file name and path.
        
        Returns:
            An NNFeatureMap object.
        """
        print(f'Opening features file: {features_file}')
        features_dir = '.ftemp/' + os.path.basename(features_file).removesuffix('.tar.gz')
        try:
            with tarfile.open(features_file, 'r:gz') as tar:
                tar.extractall(features_dir)
            features_dir += '/' + os.path.basename(features_file).removesuffix('.tar.gz')
            
            # convert json key to int
            key_to_int = lambda x: {int(k):v for k,v in x.items()}
            
            with open(features_dir + '/label_map.json', 'r') as f:
                label_map = json.loads(f.read())
            with open(features_dir + '/index_map.json', 'r') as f:
                index_map = json.loads(f.read(), object_hook=key_to_int)
            with open(features_dir + '/word_map.json', 'r') as f:
                word_map = json.loads(f.read())
            with open(features_dir + '/wordfreq_map.json', 'r') as f:
                wordfreq_map = json.loads(f.read(), object_hook=key_to_int)
            with open(features_dir + '/pos_map.json', 'r') as f:
                pos_map = json.loads(f.read())
            with open(features_dir + '/deprel_map.json', 'r') as f:
                deprel_map = json.loads(f.read())
        finally:
            shutil.rmtree('.ftemp')
        print('*FINISH*')
        
        return cls(label_map, index_map, word_map, wordfreq_map, pos_map, deprel_map)

    def save(self, features_file: str) -> None:
        """Saves an NNFeatureMap object to file.

        The files will be saved in json format in a single tarfile with gzip.

        Args:
            features_file: A string containing the file name and path.
        """
        features_file = features_file.removesuffix('.feats.tar.gz') + '.feats'
        print(f'Saving features to {features_file}.tar.gz')
        try:
            os.makedirs(features_file)
            
            with open(features_file + '/label_map.json', 'w') as f:
                f.write(json.dumps(self.label_map))
            with open(features_file + '/index_map.json', 'w') as f:
                f.write(json.dumps(self.index_map))
            with open(features_file + '/word_map.json', 'w') as f:
                f.write(json.dumps(self.word_map))
            with open(features_file + '/wordfreq_map.json', 'w') as f:
                f.write(json.dumps(self.wordfreq_map))
            with open(features_file + '/pos_map.json', 'w') as f:
                f.write(json.dumps(self.pos_map))
            with open(features_file + '/deprel_map.json', 'w') as f:
                f.write(json.dumps(self.deprel_map))
            
            with tarfile.open(features_file + '.tar.gz', 'w:gz') as tar:
                tar.add(features_file, arcname=os.path.basename(features_file))
        finally:
            shutil.rmtree(features_file)
        print('*FINISH*')


@dataclass(eq=False, slots=True)
class NNFeatures:
    """This dataclass stores the features of an instance for a neural network.
    
    Attributes:
        words: A list of strings containing words.
        pos: A list of strings containing POS.
        deprels: A list of strings containing dependency relations.
        distance: An int for the distance between the head and dependent.
    """

    words: list[str] = field(default_factory=list)
    pos: list[str] = field(default_factory=list)
    deprels: list[str] = field(default_factory=list)
    distance: int = 0
    
    def __str__(self) -> str:
        w = 'words:\n' + str(self.words)
        p = 'pos:\n' + str(self.pos)
        r = 'deprels:\n' + str(self.deprels)
        d = f'distance: {self.distance}'
        return w + '\n' + p + '\n' + r + '\n' + d