import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_reader import DataReader, Word2vecDataset
from embedders.model_pytorch import SkipGramModel, CBOW

import torch.nn as nn
import torch.autograd as autograd


class Word2VecTrainer_CBOW:
    """Object for creating walks of a dataset.

    Parameters
    ----------
    walks: list
        List containing the walks from the Word2VecWalks Object
    emb_dimension: int
        Dimensionality of the word vectors.
    batch_size: int
        Batches of examples passed to the training
    window_size: int
        Maximum distance between the current and predicted word within a sentence. 
    iterations: str
        Number of iterations (epochs) over the corpus.
    initial_lr: str
        The initial learning rate.
    min_count: str
        Ignores all entities with total frequency lower than this.
    -------
    """


    def __init__(self, walks, emb_dimension=100, batch_size=32, window_size=5, iterations=10,
                 initial_lr=0.001, min_count=0):

        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        
        raw_walks = [' '.join(s) for w in walks for s in w]
        raw_walks = ' '.join(raw_walks).split()

        vocab = set(raw_walks)
        vocab_size = len(vocab)

        self.word2id = {word: i for i, word in enumerate(vocab)}
        idx_to_word = {i: word for i, word in enumerate(vocab)}

        # Create data from the walks
        self.data = []
        for i in range(2, len(raw_walks) - 2):
            context = [raw_walks[i-2], raw_walks[i-1],
                       raw_walks[i+1], raw_walks[i+2]]
            target = raw_walks[i]
            self.data.append((context, target))

        
        # Create CBOW Model
        self.model = CBOW(vocab_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

       

    def make_context_vector(self, context, word2id):
        idxs = [word2id[w] for w in context]
        return torch.tensor(idxs, dtype=torch.long)

    def train(self):
        """Training function for training the CBOW Model."""
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.initial_lr)

        losses = []
        loss_function = nn.NLLLoss()
        for epoch in tqdm(range(self.iterations)):
            total_loss = 0
            for context, target in self.data:
                context_vector = self.make_context_vector(context, self.word2id)
                
                # Remember PyTorch accumulates gradients; zero them out
                self.model.zero_grad()
                
                nll_prob = self.model(context_vector)
                loss = loss_function(nll_prob, autograd.Variable(torch.tensor([self.word2id[target]])))
                
                # backpropagation
                loss.backward()
                # update the parameters
                optimizer.step() 
                
                total_loss += loss.item()
                
            print("Epoch {epoch} Loss: " + str(total_loss))


    def save_embedding(self, output_file_name):
        """Function for saving all embeddings."""
        self.model.save_embedding(self.idx_to_word, output_file_name)



class Word2VecTrainer_Skipgram:
    """Object for creating walks of a dataset.

    Parameters
    ----------
    walks: list
        List containing the walks from the Word2VecWalks Object
    emb_dimension: int
        Dimensionality of the word vectors.
    batch_size: int
        Batches of examples passed to the training
    window_size: int
        Maximum distance between the current and predicted word within a sentence. 
    iterations: str
        Number of iterations (epochs) over the corpus.
    initial_lr: str
        The initial learning rate.
    min_count: str
        Ignores all entities with total frequency lower than this.
    -------
    """
    def __init__(self, walks, emb_dimension=100, batch_size=32, window_size=5, iterations=10,
                 initial_lr=0.001, min_count=0):

        self.data = DataReader(walks, min_count)
        dataset = Word2vecDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.word2id = self.data.word2id
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        """Training function for training the CBOW Model."""

        for iteration in range(self.iterations):

            print("\n\n\nIteration: " + str(iteration + 1))
            optimizer = optim.SparseAdam(self.model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    optimizer.zero_grad()
                    loss = self.model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))

    
    def save_embedding(self, output_file_name):
        """Function for saving all embeddings."""
        self.model.save_embedding(self.data.id2word, output_file_name)
