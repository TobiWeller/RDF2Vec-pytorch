import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from rdf2vec import RDF2VecTransformer
from graphs import KG
from samplers import WideSampler
from walkers import HALKWalker

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from embedders.Trainer import Word2VecTrainer_CBOW, Word2VecTrainer_Skipgram

import torch.nn as nn
import torch.autograd as autograd

# Ensure the determinism of this script by initializing a pseudo-random number.
RANDOM_STATE = 42

# Class
class Word2VecWalks:
    """Object for creating walks of a dataset.

    Parameters
    ----------
    train_data: str
        Link to TSV File containing the entities for training with the corresponding label.
    test_data: str
        Link to TSV File containing the entities for testing with the corresponding label.
    label: str
        Column of the train and test data which contains the label to be predicted.
    -------
    """


    def __init__(self, train_data, test_data, label):
        self.train_data = pd.read_csv(train_data, sep="\t")
        self.test_data = pd.read_csv(test_data, sep="\t")
        

        self.train_entities = [entity for entity in self.train_data["bond"]]
        self.train_labels = list(self.train_data[label])

        self.test_entities = [entity for entity in self.test_data["bond"]]
        self.test_labels = list(self.test_data[label])

        self.entities = self.train_entities + self.test_entities
        self.labels = self.train_labels + self.test_labels




    def get_walks(self, kg_file, skip_predicates, literals):
        """Function for creating walks of a dataset.

        Parameters
        ----------
        kg_file: str
            Link to RDF Graph for which the walks should be created.
        skip_predicates: set
            Ignore property while creating the walks (usually the label in the graph we want to predict).
        literals: list
            List of predicate chains to get the literals.
        -------
        """

        rdf2vec_walks = RDF2VecTransformer(
            # Extract all walks with a maximum depth of 2 for each entity using two
            # processes and use a random state to ensure that the same walks are
            # generated for the entities without hashing as MUTAG is a short KG.
            walkers=[
                HALKWalker(
                    2, # max_depths
                    None, # max_walks
                    n_jobs=2,
                    sampler=WideSampler(),
                    random_state=RANDOM_STATE,
                    md5_bytes=None,
                )
            ],
            verbose=1,
        )

        walks = rdf2vec_walks.get_walks(KG(
                kg_file,
                skip_predicates=skip_predicates,
                literals=literals
            ),
            self.entities)

        return walks

    def get_data(self):
        """Function for retrieving the entities and labels for evaluation."""

        return self.train_entities, self.test_entities, self.entities, self.train_labels, self.test_labels, self.labels


if __name__ == '__main__':
    #generate walks for MUTAG Dataset
    walks_obj = Word2VecWalks('./data/mutag/train.tsv', './data/mutag/test.tsv', 'label_mutagenic')
    walks = walks_obj.get_walks('./data/mutag/mutag.owl', {'http://dl-learner.org/carcinogenesis#isMutagenic'}, [['http://dl-learner.org/carcinogenesis#hasBond', 'http://dl-learner.org/carcinogenesis#inBond'], ['http://dl-learner.org/carcinogenesis#hasAtom', 'http://dl-learner.org/carcinogenesis#charge']])
    train_entities, test_entities, entities, train_labels, test_labels, labels = walks_obj.get_data()
    

    # Create Skipgram or CBOW Model and train it.
    #w2v = Word2VecTrainer_CBOW(walks=walks, output_file="out.vec", iterations=1, min_count=0)
    w2v = Word2VecTrainer_Skipgram(walks=walks, iterations=10, min_count=0)
    w2v.train()
    # optional: Store Embeddings
    #w2v.save_embedding('./data/mutag/word2vec.vec')


    # Get Embeddings for train and test set
    train_embeddings = w2v.model.get_embedding(w2v.word2id, train_entities)
    test_embeddings = w2v.model.get_embedding(w2v.word2id, test_entities)

    # Fit a Support Vector Machine on train embeddings and pick the best
    # C-parameters (regularization strength).
    clf = GridSearchCV(
        SVC(random_state=RANDOM_STATE), {"C": [10 ** i for i in range(-3, 4)]}
    )
    clf.fit(train_embeddings, train_labels)

    # Evaluate the Support Vector Machine on test embeddings.
    predictions = clf.predict(test_embeddings)
    print(
        f"Predicted {len(test_entities)} entities with an accuracy of "
        + f"{accuracy_score(test_labels, predictions) * 100 :.4f}%"
    )
    print(f"Confusion Matrix ([[TN, FP], [FN, TP]]):")
    print(confusion_matrix(test_labels, predictions))

    # Reduce the dimensions of entity embeddings to represent them in a 2D plane.
    X_tsne = TSNE(random_state=RANDOM_STATE).fit_transform(
        train_embeddings + test_embeddings
    )

    # Define the color map.
    colors = ["r", "g"]
    color_map = {}
    for i, label in enumerate(set(labels)):
        color_map[label] = colors[i]

    # Set the graph with a certain size.
    plt.figure(figsize=(10, 4))

    # Plot the train embeddings.
    plt.scatter(
        X_tsne[: len(train_entities), 0],
        X_tsne[: len(train_entities), 1],
        edgecolors=[color_map[i] for i in labels[: len(train_entities)]],
        facecolors=[color_map[i] for i in labels[: len(train_entities)]],
    )

    # Plot the test embeddings.
    plt.scatter(
        X_tsne[len(train_entities) :, 0],
        X_tsne[len(train_entities) :, 1],
        edgecolors=[color_map[i] for i in labels[len(train_entities) :]],
        facecolors="none",
    )

    # Annotate few points.
    plt.annotate(
        entities[25].split("/")[-1],
        xy=(X_tsne[25, 0], X_tsne[25, 1]),
        xycoords="data",
        xytext=(0.01, 0.0),
        fontsize=8,
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", facecolor="black"),
    )
    plt.annotate(
        entities[35].split("/")[-1],
        xy=(X_tsne[35, 0], X_tsne[35, 1]),
        xycoords="data",
        xytext=(0.4, 0.0),
        fontsize=8,
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", facecolor="black"),
    )

    # Create a legend.
    plt.scatter([], [], edgecolors="r", facecolors="r", label="train -")
    plt.scatter([], [], edgecolors="g", facecolors="g", label="train +")
    plt.scatter([], [], edgecolors="r", facecolors="none", label="test -")
    plt.scatter([], [], edgecolors="g", facecolors="none", label="test +")
    plt.legend(loc="upper right", ncol=2)

    # Display the graph with a title, removing the axes for
    # better readability.
    plt.title("pyRDF2Vec", fontsize=32)
    plt.axis("off")
    plt.show()