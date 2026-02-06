# Importing all the required libraries
import torch
import torch.nn as nn
from utils import LayerNorm, ResidualConnection, FeedForwardNetwork, MultiHeadAttentionLayer

class EncoderBlock(nn.Module):
    """
    A single layer of encoder.
    """
    def __init__(self, d_model: int, self_attention_layer: MultiHeadAttentionLayer, feed_forward_network: FeedForwardNetwork, dropout: float) -> None:
        """
        Args:
            d_model (int): The dimension of each embedding vector.
            self_attention_layer (MultiHeadAttentionLayer): Layer implementing self attention.
            feed_forward_network (FeedForwardNetwork): Layer implementing feedforward neural network.
            dropout (float): Dropout value.
        """
        super().__init__()
        self.self_attention_layer = self_attention_layer
        self.feed_forward_network = feed_forward_network
        self.residual_connection_1 = ResidualConnection(d_model, dropout)
        self.residual_connection_2 = ResidualConnection(d_model, dropout)

    def forward(self, x, src_mask) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Matrix of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): Source mask.

        Returns:
            x (torch.Tensor): Resultant matrix of shape (batch_size, seq_len, d_model).
        """
        # We need to use lambda because we need to pass a function as parameter
        x = self.residual_connection_1(x, lambda x: self.self_attention_layer(x, x, x, src_mask))
        x = self.residual_connection_2(x, lambda x: self.feed_forward_network(x))
        return x

class Encoder(nn.Module):
    """
    Assembles all the encoder layers.
    """
    def __init__(self, num_encoders: int, d_model: int, num_heads: int, hidden_size: int, seq_len: int, pe_method: str = "sinusoidal", dropout: float = 0, pe_object = None) -> None:
        """
        Args:
            num_encoders (int): Number of encoders.
            d_model (int): The dimension of each embedding vector.
            num_heads (int): Number of heads.
            hidden_size (int): The size of hidden layer in feedforward network.
            seq_len (int): length of the sequence
            pe_method (str): positional encoding method
            dropout (float): Dropout value.
        """
        super().__init__()
        self.num_encoders = num_encoders
        self.norm = LayerNorm(d_model)

        encoder_layers = []
        for i in range(num_encoders):
            self_attention_layer = MultiHeadAttentionLayer(d_model, num_heads, seq_len, dropout, pe_method, pe_object)
            feed_forward_network = FeedForwardNetwork(d_model, hidden_size, dropout)
            encoder_layer = EncoderBlock(d_model, self_attention_layer, feed_forward_network, dropout)
            encoder_layers.append(encoder_layer)
        
        # We need to make encoder_layers as a ModuleList
        self.encoder_layers = nn.ModuleList(encoder_layers)

    def forward(self, x, src_mask) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Matrix of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): Source mask of shape (1, 1, 1, seq_len) for dealing with padding tokens

        Returns:
            x (torch.Tensor): Resultant matrix of shape (batch_size, seq_len, d_model).
        """
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return self.norm(x)