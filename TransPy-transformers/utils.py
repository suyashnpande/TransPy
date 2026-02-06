import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    """
    Converts tokens to dense vector representations.
    """
    def __init__(self, vocab_size: int, d_model: int) -> None:
        """
        Args:
            vocab_size (int): number of words in the vocabulory.
            d_model (int): dimension of each embedding vector.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding_layer = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): matrix of shape (batch_size, seq_len).
        
        Returns:
            embeddings (torch.Tensor): embedding matrix of shape (batch_size, seq_len, d_model).
        """
        embeddings = self.embedding_layer(x)
        return embeddings

class LayerNorm(nn.Module):
    """
    Normalizes the data across dimensions.
    """
    def __init__(self, d_model: int) -> None:
        """
        Args:
            d_model (int): dimension of each embedding vector.
        """
        super().__init__()
        # We need two trainable param vector alpha and beta for each dimension
        self.alpha = nn.Parameter(torch.ones(d_model).unsqueeze(0).unsqueeze(0))    
        self.beta = nn.Parameter(torch.zeros(d_model).unsqueeze(0).unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input of shape (batch_size, seq_len, d_model).
        
        Returns:
            x (torch.Tensor): layer normalized matrix of shape (batch_size, seq_len, d_model).
        """
        epsilon = 1e-5    # Using epsilon to avoid 0 in denominator

        # The input is (batch_size, seq_len, d_model)
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)

        x = self.alpha*((x - mean)/(std + epsilon)) + self.beta

        return x

class FeedForwardNetwork(nn.Module):
    """
    Introduces non-linearity in each layer.
    """
    def __init__(self, d_model: int, hidden_size: int, dropout: float) -> None:
        """
        Args:
            d_model (int): dimension of each embedding vector.
            hidden_size (int): size of hidden layer in feedforward neural network.
            dropout (float): dropout value.
        """
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size

        # Defining the weights 
        self.w1 = nn.Linear(d_model, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(hidden_size, d_model)
    
    def forward(self, x: torch.Tensor) ->  torch.Tensor:
        """
        Args:
            x (torch.Tensor): input of shape (batch_size, seq_len, d_model).
        
        Returns:
            x (torch.Tensor): transformed matrix of shape (batch_size, seq_len, d_model).
        """
        x = self.relu(self.w1(x))
        x = self.dropout(x)
        x = self.w2(x)
        return x

class ResidualConnection(nn.Module):
    """
    Adds a residual connection around a sublayer.
    """
    def __init__(self, d_model: int, dropout: float) -> None:
        """
        Args:
            d_model (int): dimension of each embedding vector.
            dropout (float): dropout value.
        """
        super().__init__()
        self.layernorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input of shape (batch_size, seq_len, d_model).
            sublayer (callable): function like self attention or feed forward network.
        
        Returns:
            x (torch.Tensor): resultant matrix of shape (batch_size, seq_len, d_model).
        """
        sublayer_output = sublayer(self.layernorm(x))    # Pre-Norm
        return x + self.dropout(sublayer_output)    # The residual path does not get dropout 

class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional information to the embedding matrix.
    """
    def __init__(self, seq_len: int, d_model: int) -> None:
        """
        Args:
            seq_len (int): number of tokens in the sequence.
            d_model (int): dimension of each embedding vector.
        """
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))    # Taking log and exponential is more stable

        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)

        pe = pe.unsqueeze(0)    # This makes it (1, seq_len, d_model) for broadcasting later
        self.register_buffer('pe', pe)    # We need to register pe as buffer so that it can be moved to GPU during training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): embedding matrix of shape (batch_size, seq_len, d_model).
        
        Returns:
            x (torch.Tensor): position encoded embedding matrix of shape (batch_size, seq_len, d_model).
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)    # (batch_size, seq_len, d_model) + (1, seq_len, d_model)
        return x
    
class RotaryPositionalEncoding(nn.Module):
    """
    Adds RoPE positional information to the embedding matrix.
    """
    def __init__(self, seq_len: int, d_model: int, num_heads: int):
        """
        Args:
            seq_len (int): number of tokens in the sequence.
            d_model (int): dimension of each embedding vector.
            num_heads (int): number of heads in each layer
        """
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        if self.d_k % 2 != 0:
            raise ValueError("Head dimension must be even for RoPE.")

        theta = 1.0 / (10000 ** (torch.arange(0, self.d_k, 2).float() / self.d_k))  # (d_k // 2)
        seq_idx = torch.arange(seq_len).float()  # (seq_len)

        freqs = seq_idx[:, None] * theta[None, :]  # (seq_len, d_k // 2)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)

        self.register_buffer("sin", sin[None, None, :, :], persistent=False)
        self.register_buffer("cos", cos[None, None, :, :], persistent=False)

    def get_rotary_emb(self, x: torch.Tensor):
        """
        Args:
            x (torch.tensor): Tensor of shape (batch_size, num_heads, seq_len, d_k)
        Returns:
            sin, cos (torch.tensor): (1, 1, seq_len, d_k // 2)
        """
        return self.sin[:, :, :x.size(2), :], self.cos[:, :, :x.size(2), :]    # Only use portion required by actual seq_len of x

class RelativePositionBias(nn.Module):
    """
    Adds relative bias positional information to the embedding matrix.
    """
    def __init__(self, seq_len: int, num_heads: int):
        """
        Args:
            seq_len (int): number of tokens in the sequence.
            d_model (int): dimension of each embedding vector.
        """
        super().__init__()
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.relative_attention_bias = nn.Embedding(2 * seq_len - 1, num_heads)    # We learn different relative bias for each head

    def forward(self, qlen: int, klen: int):
        context_position = torch.arange(qlen, device=self.relative_attention_bias.weight.device)[:, None]
        memory_position = torch.arange(klen, device=self.relative_attention_bias.weight.device)[None, :]
        relative_position = memory_position - context_position  # (qlen, klen)

        relative_position += (self.seq_len - 1)  # shift to positive range

        values = self.relative_attention_bias(relative_position)  # This replaces the relative distance with the respective embedding
        return values.permute(2, 0, 1).unsqueeze(0)  
    
class MultiHeadAttentionLayer(nn.Module):
    """
    Implements self attention, masked attention and cross attention.
    """
    def __init__(self, d_model: int, num_heads: int, seq_len: int, dropout: float = 0, pe_method: str = "sinusoidal", pe_object=None) -> None:
        """
        Args:
            d_model (int): dimension of each embedding vector.
            num_heads (int): number of heads in a multihead attention layer.
            seq_len (int): length of the sequence
            dropout (float): dropout value.
            pe_method (str): positional encoding method (sinusoidal / relative_bias / rope).
            pe_object (Object): RoPE object
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.pe_method = pe_method

        if (d_model % num_heads != 0):
            print(f"d_model {d_model} should be a multiple of num_heads {num_heads}")
        
        self.d_k = d_model // num_heads

        self.w_key = nn.Linear(d_model, d_model)
        self.w_query = nn.Linear(d_model, d_model)
        self.w_value = nn.Linear(d_model, d_model)
        self.w_output = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        if (self.pe_method == "rope"):
            self.rope = pe_object
        elif (self.pe_method == "relative_bias"):
            self.relative_bias = pe_object

    def apply_rope(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        """
        Applies Rotary Positional Embedding (RoPE) to the last dimension of the input tensor.

        Args:
            x: Tensor of shape (batch_size, num_heads, seq_len, head_dim)
            sin: Tensor of shape (seq_len, 1, head_dim // 2)
            cos: Tensor of shape (seq_len, 1, head_dim // 2)

        Returns:
            Tensor of same shape as x, with RoPE applied.
        """
        x1 = x[..., ::2]  # even dims
        x2 = x[..., 1::2]  # odd dims

        # Apply rotation
        x_rotated_even = x1 * cos - x2 * sin
        x_rotated_odd  = x1 * sin + x2 * cos

        # Interleave even and odd dimensions
        x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1)
        x_rotated = x_rotated.flatten(-2)

        return x_rotated


    def attention(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            key (torch.Tensor): input of shape (batch_size, num_heads, seq_len, d_k).
            query (torch.Tensor): input of shape (batch_size, num_heads, seq_len, d_k).
            value (torch.Tensor): input of shape (batch_size, num_heads, seq_len, d_k).
            mask (torch.Tensor): source or target mask depending on attention type
        
        Returns:
            x (torch.Tensor): Resultant matrix of shape (batch_size, num_heads, seq_len, d_k).
        """
        # We want to take the dot products between key transpose and query, but we need to adjust for (batch_size, seq_len) in both
        attention_scores = (torch.matmul(query, key.transpose(-2,-1))) / math.sqrt(self.d_k)

        if (self.pe_method == "relative_bias"):
            attention_scores = attention_scores + self.re_bias

        if mask is not None:
            # Replacing masked attention scores with -inf
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_scores = attention_scores.softmax(dim=-1)    # Here, -1 dim is important because we take softmax across key sequence

        # Applying attention dropout
        attention_scores = self.dropout(attention_scores)
        attention_scores = torch.matmul(attention_scores, value)

        return attention_scores
    
    def forward(self, k, q, v, mask = None) -> torch.Tensor:
        """
        Args:
            k (torch.Tensor): Input matrix of shape (batch_size, seq_len, d_model).
            q (torch.Tensor): Input matrix of shape (batch_size, seq_len, d_model).
            v (torch.Tensor): Input matrix of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Mask of shape (batch_size, num_heads, qlen, klen).
        
        Returns:
            x (torch.Tensor): Resultant matrix of shape (batch_size, seq_len, d_model).
        """
        key = self.w_key(k)
        query = self.w_query(q)
        value = self.w_value(v)

        # We need to split the input on embedding axis into num_heads parts
        # The idea is to convert (batch_size, seq_len, d_model) to (batch_size, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k)
        key = key.transpose(1, 2)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k)
        query = query.transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        if self.pe_method == "rope":
            sin, cos = self.rope.get_rotary_emb(query)
            query = self.apply_rope(query, sin, cos)

            sin, cos = self.rope.get_rotary_emb(key)
            key = self.apply_rope(key, sin, cos)
        
        if self.pe_method == "relative_bias":
            self.re_bias = self.relative_bias(query.size(2), key.size(2))

        x = self.attention(key, query, value, mask)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads*self.d_k)    # Concatenation

        output = self.w_output(x)

        return output

def causal_mask(seq_len):
    """
    Causal mask ensures that positions above the main diagonal are masked out, preventing the model from attending to future tokens during decoding.

    Args:
        seq_len (int): Length of the sequence to generate the mask for.

    Returns:
        torch.BoolTensor: A mask tensor of shape (1, seq_len, seq_len) where True values indicate allowed positions and False values indicate masked positions.
    """
    mask = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).type(torch.int)    # All the values above diagonal will be 1
    return mask == 0    # All the values above diagonal will be 0, and rest will be 1