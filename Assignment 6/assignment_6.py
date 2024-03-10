import numpy as np
from pathlib import Path
from typing import Tuple


class Node:
    """ Node class used to build the decision tree"""

    def __init__(self):
        self.children = {}
        self.parent = None
        self.attribute = None
        self.value = None

    def classify(self, example):
        if self.value is not None:
            return self.value
        if example[self.attribute] not in self.children:
            # Diagnostic print statement to log when an unknown attribute value is encountered
            print(f"Unknown attribute value encountered: {example[self.attribute]} for attribute {self.attribute}")
            # return the plurality value of the parent examples
        return self.children[example[self.attribute]].classify(example)


def plurality_value(examples: np.ndarray) -> int:
    """Implements the PLURALITY-VALUE (Figure 19.5)"""
    labels = examples[:, -1]
    value, count = 0, 0
    for label in np.unique(labels):
        label_count = np.count_nonzero(labels == label)
        if label_count > count:
            value = label
            count = label_count

    return value


def entropy(examples: np.ndarray) -> float:
    """Calculates the entropy of the given examples"""
    labels = examples[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))


def information_gain(examples: np.ndarray, attribute: int) -> float:
    """Calculates the information gain of an attribute"""
    initial_entropy = entropy(examples)
    values, counts = np.unique(examples[:, attribute], return_counts=True)
    weighted_entropy = sum(
        (counts[i] / counts.sum()) * entropy(examples[examples[:, attribute] == v])
        for i, v in enumerate(values)
    )
    return initial_entropy - weighted_entropy


def importance(attributes: np.ndarray, examples: np.ndarray, measure: str) -> int:
    """Calculates the importance of each attribute"""
    if measure == "random":
        return np.random.choice(attributes)
    elif measure == "information_gain":
        gains = np.array([information_gain(examples, attribute) for attribute in attributes])
        return attributes[np.argmax(gains)]


def learn_decision_tree(examples: np.ndarray, attributes: np.ndarray, parent_examples: np.ndarray,
                        parent: Node, branch_value: int, measure: str) -> Node:
    """Learns a decision tree from the given examples"""
    node = Node()

    if parent is not None:
        parent.children[branch_value] = node
        node.parent = parent

    # If examples is empty then return a leaf node with the plurality value of the parent examples
    if examples.size == 0:
        node.value = plurality_value(parent_examples)
        return node

    # If all examples have the same classification, return a leaf node with that classification
    if np.unique(examples[:, -1]).size == 1:
        node.value = examples[0, -1]
        return node

    # If attributes is empty, return a leaf node with the plurality value of the examples
    if attributes.size == 0:
        node.value = plurality_value(examples)
        return node

    # Choose the attribute that best classifies the examples or a random one
    best_attr = importance(attributes, examples, measure)
    node.attribute = best_attr
    possible_values = (1,2)
    # iterate over the values of the best attribute and create a subtree for each value
    for value in possible_values:
        subtree = learn_decision_tree(
            # examples where the best attribute is equal to the value
            examples[examples[:, best_attr] == value],
            # remove the best attribute from the list of attributes
            np.delete(attributes, np.where(attributes == best_attr)),
            # parent examples
            examples,
            # parent
            node,
            # branch
            value,
            measure
        )
        node.children[value] = subtree

    return node


def accuracy(tree: Node, examples: np.ndarray) -> float:
    """ Calculates accuracy of tree on examples """
    correct = 0
    for example in examples:
        pred = tree.classify(example[:-1])
        correct += pred == example[-1]

    return correct / examples.shape[0]


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Load the data for the assignment,
    Assumes that the data files is in the same folder as the script"""
    with (Path.cwd() / "train.csv").open("r") as f:
        train = np.genfromtxt(f, delimiter=",", dtype=int)
    with (Path.cwd() / "test.csv").open("r") as f:
        test = np.genfromtxt(f, delimiter=",", dtype=int)
    return train, test


if __name__ == '__main__':
    train, test = load_data()

    # information_gain or random
    random = "random"
    information_gain = "information_gain"
    measure = random


    for i in range(10):
        tree = learn_decision_tree(examples=train,
                                   attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int),
                                   parent_examples=None,
                                   parent=None,
                                   branch_value=None,
                                   measure=measure)

        print(f"Training Accuracy {accuracy(tree, train)}")
        print(f"Test Accuracy {accuracy(tree, test)}")
