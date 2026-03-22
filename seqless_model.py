"""
Sequenceless Micro16S Model

This module implements SequencelessMicro16S, a proof-of-concept model that learns
a lookup table of embeddings for the training set rather than learning how to embed
DNA sequences. This removes the challenges of DNA sequence processing and focuses
purely on the mining algorithm, loss functions, and embedding space optimization.

The model is simply a learnable embedding matrix where each row is the embedding
for one training sequence. During training, the model receives sequence indices
(0 to n_train-1) instead of DNA sequences, and returns the corresponding embeddings.

Architecture:
    - Input: Sequence indices (batch_size,) as integers in [0, n_train-1]
    - Embedding Layer: (n_train, embed_dims) learnable parameters
    - Output: L2-normalized embeddings (batch_size, embed_dims)

This model is compatible with the existing training loop, loss functions, and
mining algorithms. It skips quick testing since there's no generalization to test.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequencelessMicro16S(nn.Module):
    """
    A sequenceless model that learns fixed embeddings for each training sequence.
    
    This is a proof-of-concept model that replaces the full Micro16S architecture
    with a simple embedding lookup table. Each training sequence gets one learnable
    embedding vector, updated during training via the standard loss functions.
    
    Args:
        n_train_sequences (int): Number of sequences in the training set
        embed_dims (int): Dimension of the embedding space
        name (str): Model name for logging/saving (optional)
    """
    
    def __init__(self, n_train_sequences, embed_dims, name="sequenceless_m16s"):
        super(SequencelessMicro16S, self).__init__()
        
        # Store configuration
        self.n_train_sequences = n_train_sequences
        self.embed_dims = embed_dims
        self.name = name
        
        # Learnable embedding matrix: one row per training sequence
        # Initialized with Xavier uniform to match typical transformer initialization
        self.embeddings = nn.Embedding(n_train_sequences, embed_dims)
        nn.init.xavier_uniform_(self.embeddings.weight)
    
    def forward(self, indices):
        """
        Look up embeddings for the given sequence indices.
        
        Args:
            indices (torch.Tensor): Sequence indices
                - Shape: (batch_size,) for single indices
                - Shape: (batch_size, n_seqs_per_sample) for pairs/triplets
                - dtype: torch.long (int64)
                - Values in [0, n_train_sequences-1]
        
        Returns:
            embeddings (torch.Tensor): L2-normalized embeddings
                - Shape: (batch_size, embed_dims) for single indices
                - Shape: (batch_size, n_seqs_per_sample, embed_dims) for pairs/triplets
        """
        # Look up embeddings from the embedding matrix
        embeddings = self.embeddings(indices)
        # embeddings.shape: (batch_size, ..., embed_dims)
        
        # L2-normalize the embeddings
        # This matches the behavior of the standard Micro16S model which outputs normalized embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)
        
        return embeddings
    
    def save_model(self, path):
        """
        Save model state dict to file.
        
        Note: In sequenceless mode, we don't typically need to save models since
        they don't generalize beyond the training set. This method is provided
        for completeness but may not be used.
        
        Args:
            path (str): Path to save the model state dict
        """
        # Save the model state dict
        torch.save({
            'model_state_dict': self.state_dict(),
            'n_train_sequences': self.n_train_sequences,
            'embed_dims': self.embed_dims,
            'name': self.name,
        }, path)
    
    @staticmethod
    def load_model(path, device='cpu'):
        """
        Load a saved SequencelessMicro16S model from file.
        
        Args:
            path (str): Path to the saved model
            device (str or torch.device): Device to load the model onto
        
        Returns:
            model (SequencelessMicro16S): Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Create model with saved configuration
        model = SequencelessMicro16S(
            n_train_sequences=checkpoint['n_train_sequences'],
            embed_dims=checkpoint['embed_dims'],
            name=checkpoint.get('name', 'sequenceless_m16s')
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        return model
