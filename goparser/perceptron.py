"""This module contains perceptron algorithms."""
from dataclasses import dataclass

import numpy as np


BIAS_INDEX = -1
"""Int constant for the perceptron bias index."""


@dataclass
class Score:
    """A data class for storing the accuracy score.
    
    Attributes:
        correct: An int with the number of correctly predicted instances.
        total: An int with the total number of instances.
    """

    correct: int = 0
    total: int = 0

    def get_score(self) -> float:
        """Returns the accuracy score."""
        return self.correct / self.total

    def reset_score(self) -> None:
        """Resets the accuracy score."""
        self.correct = 0
        self.total = 0


class Matrix:
    """This class contains the weight matrix for the perceptron.
    
    Attributes:
        w: A m x n NumPy array for the weight matrix.
    """

    __slots__ = ('w',)

    def __init__(self, w: np.ndarray) -> None:
        self.w = w


class Weights(Matrix):
    """This class contains methods for importing and exporting weights,
    and for predicting using the weights."""

    @classmethod
    def import_weights(cls, weights_file: str):
        """Import a saved weights file."""
        return cls(np.load(weights_file))

    def export_weights(self, weights_file: str) -> None:
        """Export a weights file to disk."""
        np.save(weights_file, self.w)

    def predict(self, x: list[int]) -> np.ndarray:
        """Returns the vector of scores given a list of feature indices
        and the correct class label.
        
        Args:
            x: A list of feature indices.
        
        Returns:
            A NumPy array of scores.
        """
        φ_x = np.zeros(self.w.shape[1], dtype=np.int64)
        φ_x[x] = 1
        φ_x[BIAS_INDEX] = 1
        return np.dot(self.w, φ_x)


class StandardMulticlassPerceptron(Matrix, Score):
    """This class implements a standard multiclass perceptron."""

    def __init__(self, m: int, n: int) -> None:
        """Inits a StandardMulticlassPerceptron instance given the number of rows
        (features) and the number of columns (class labels).
        
        Args:
            m: An int for the number of rows (features).
            n: An int for the number of columns (class labels).
        """
        super().__init__(np.zeros((m, n + 1), dtype=np.int64))

    def train(self, x: list[int], y: int) -> None:
        """Trains the perceptron on an example given a list of feature indices
        and the correct class label.
        
        Args:
            x: A list of feature indices.
            y: An int for the correct class label.
        """
        φ_x = np.zeros(self.w.shape[1], dtype=np.int64)
        φ_x[x] = 1
        φ_x[BIAS_INDEX] = 1
        ŷ = np.argmax(np.dot(self.w, φ_x))
        if ŷ != y:
            self.w[y] += φ_x
            self.w[ŷ] -= φ_x
        else:
            self.correct += 1
        self.total += 1

    def get_weights(self) -> Weights:
        """Returns the weights of the perceptron as a Weights object."""
        return Weights(self.w)


class AveragedMulticlassPerceptron(Matrix, Score):
    """This class implements an averaged multiclass perceptron.
    
    Attributes:
        u: A m x n NumPy array for the cached weight matrix.
        q: An int for the example counter.
    """

    __slots__ = ('u', 'q')

    def __init__(self, m: int, n: int) -> None:
        """Inits a AveragedMulticlassPerceptron instance given the number of rows
        (features) and the number of columns (class labels).
        
        Args:
            m: An int for the number of rows (features).
            n: An int for the number of columns (class labels).
        """
        super().__init__(np.zeros((m, n + 1), dtype=np.int64))
        self.u = np.copy(self.w)
        self.q = 0

    def train(self, x: list[int], y: int) -> None:
        """Trains the perceptron on an example given a list of feature indices
        and the correct class label.
        
        Args:
            x: A list of feature indices.
            y: An int for the correct class label.
        """
        self.q += 1
        φ_x = np.zeros(self.w.shape[1], dtype=np.int64)
        φ_x[x] = 1
        φ_x[BIAS_INDEX] = 1
        ŷ = np.argmax(np.dot(self.w, φ_x))
        if ŷ != y:
            self.w[y] += φ_x
            self.w[ŷ] -= φ_x
            self.u[y] += np.dot(self.q, φ_x)
            self.u[ŷ] -= np.dot(self.q, φ_x)
        else:
            self.correct += 1
        self.total += 1

    def get_weights(self) -> Weights:
        """Returns the weights of the perceptron as a Weights object."""
        return Weights(self.w - np.dot((1 / self.q), self.u))


MulticlassPerceptron = StandardMulticlassPerceptron | AveragedMulticlassPerceptron
"""Type alias for perceptron classes."""