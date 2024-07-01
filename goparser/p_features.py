"""This module contains dataclasses for storing and mapping
features and class labels for a perceptron.
"""
from dataclasses import dataclass, field
import json
import os
import shutil
import tarfile
from typing import Any, Callable


@dataclass(eq=False, slots=True)
class PFeatureMap:
    """This dataclass contains the feature-to-index maps.

    Attributes:
        template: A function that returns a list of feature strings.
        feature_map: A dictionary mapping features to indices.
        label_map: A dictionary mapping class labels to indices.
        index_map: A dictionary mapping indices to class labels.
        freeze: A bool to freeze the FeatureMap.
            Default: False.
    """

    template: Callable[[Any], list[str]]
    feature_map: dict[str, int] = field(default_factory=dict)
    label_map: dict[str, int] = field(default_factory=dict)
    index_map: dict[int, str] = field(default_factory=dict)
    freeze: bool = False
    
    @classmethod
    def open(cls, template: Callable[[Any], list[str]], features_file: str):
        """Opens an PFeatureMap file.

        Args:
            template: A function that returns a list of feature strings.
            features_file: A string containing the file name and path.
        
        Returns:
            A PFeatures object.
        """
        print(f'Opening features file: {features_file}')
        features_dir = '.ftemp/' + os.path.basename(features_file).removesuffix('.tar.gz')
        try:
            with tarfile.open(features_file, 'r:gz') as tar:
                tar.extractall(features_dir)
            features_dir += '/' + os.path.basename(features_file).removesuffix('.tar.gz')
            
            # convert json key to int
            key_to_int = lambda x: {int(k):v for k,v in x.items()}
            
            with open(features_dir + '/word_map.json', 'r') as f:
                feature_map = json.loads(f.read())
            with open(features_dir + '/label_map.json', 'r') as f:
                label_map = json.loads(f.read())
            with open(features_dir + '/index_map.json', 'r') as f:
                index_map = json.loads(f.read(), object_hook=key_to_int)
        finally:
            shutil.rmtree('.ftemp')
        print('*FINISH*')
        
        return cls(template, feature_map, label_map, index_map, freeze=True)

    def save(self, features_file: str) -> None:
        """Saves an FeatureMap object to file.

        The files will be saved in json format in a single tarfile with gzip.

        Args:
            features_file: A string containing the file name and path.
        """
        features_file = features_file.removesuffix('.feats.tar.gz') + '.feats'
        print(f'Saving features to {features_file}.tar.gz')
        try:
            os.makedirs(features_file)
            
            with open(features_file + '/feature_map.json', 'w') as f:
                f.write(json.dumps(self.feature_map))
            with open(features_file + '/label_map.json', 'w') as f:
                f.write(json.dumps(self.label_map))
            with open(features_file + '/index_map.json', 'w') as f:
                f.write(json.dumps(self.index_map))
            
            with tarfile.open(features_file + '.tar.gz', 'w:gz') as tar:
                tar.add(features_file, arcname=os.path.basename(features_file))
        finally:
            shutil.rmtree(features_file)
        print('*FINISH*')
    
    def get_feature_vector(self, *args, **kwargs) -> list[int]:
        """Returns a list of feature indices."""
        feature_vector = list[int]()
        feats = self.template(*args, **kwargs)
        for feat in feats:
            if feat not in self.feature_map and not self.freeze:
                self.feature_map[feat] = len(self.feature_map)
            index = self.feature_map.get(feat)
            if index is not None:
                feature_vector.append(index)
        return feature_vector


@dataclass(eq=False, slots=True)
class PFeatures:
    """This dataclass stores the features of an instance for a perceptron.
    
    Attributes:
        label: An int index of the correct label.
        feats: A list of int indices corresponding to instance features.
    """
    
    label: int
    feats: list[int] = field(default_factory=list)
    
    def __str__(self) -> str:
        l = f'label: {self.label}'
        f = 'feats:\n' + str(self.feats)
        return l + '\n' + f