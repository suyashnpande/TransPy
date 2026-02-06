# Importing all the required libraries
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import LayerNorm, ResidualConnection, FeedForwardNetwork, MultiHeadAttentionLayer, causal_mask

class DecoderBlock(nn.Module):
    """
    Implements a single layer of decoder..
    """
    def __init__(self, d_model: int, masked_attention_layer: MultiHeadAttentionLayer, cross_attention_layer: MultiHeadAttentionLayer, feed_forward_network: FeedForwardNetwork, dropout: float) -> None:
        """
        Args:
            d_model (int): The dimension of each embedding vector.
            masked_attention_layer (MultiHeadAttentionLayer): Layer implementing masked self attention.
            cross_attention_layer (MultiHeadAttentionLayer): Layer implementing cross attention.
            feed_forward_network (FeedForwardNetwork): Layer implementing feedforward neural network.
            dropout (float): Dropout value.
        """
        super().__init__()
        self.masked_attention_layer = masked_attention_layer
        self.cross_attention_layer = cross_attention_layer
        self.feed_forward_network = feed_forward_network
        self.residual_connection_1 = ResidualConnection(d_model, dropout)
        self.residual_connection_2 = ResidualConnection(d_model, dropout)
        self.residual_connection_3 = ResidualConnection(d_model, dropout)
    
    def forward(self, dec_input, tgt_mask, enc_output, src_mask) -> torch.Tensor:
        """
        Args:
            dec_input (torch.Tensor): Matrix of shape (batch_size, seq_len, d_model).
            tgt_mask (torch.Tensor): Target mask.
            enc_output (torch.Tensor): Output from the encoder of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): Source mask.

        Returns:
            x (torch.Tensor): Resultant matrix of shape (batch_size, seq_len, d_model).
        """
        x = self.residual_connection_1(dec_input, lambda x: self.masked_attention_layer(dec_input, dec_input, dec_input, tgt_mask))
        x = self.residual_connection_2(x, lambda x: self.cross_attention_layer(enc_output, x, enc_output, src_mask))    # The encoder output acts as key and value
        x = self.residual_connection_3(x, lambda x: self.feed_forward_network(x))
        return x

class Decoder(nn.Module):
    """
    Assembles all the decoder layers.
    """
    def __init__(self, num_decoders: int, d_model: int, num_heads: int, hidden_size: int, seq_len: int, pe_method: str = None, dropout: float = 0, pe_object = None) -> None:
        """
        Args:
            num_decoders (int): Number of decoders.
            d_model (int): The dimension of each embedding vector.
            num_heads (int): Number of heads.
            hidden_size (int): The size of hidden layer in feedforward network.
            seq_len (int): length of the sequence
            pe_method (str): positional encoding method
            dropout (float): Dropout value.
        """
        super().__init__()
        self.num_decoders = num_decoders
        self.norm = LayerNorm(d_model)

        decoder_layers = []
        for i in range(num_decoders):
            masked_attention_layer = MultiHeadAttentionLayer(d_model, num_heads, seq_len, dropout, pe_method, pe_object)
            cross_attention_layer = MultiHeadAttentionLayer(d_model, num_heads, seq_len, dropout, pe_method, pe_object)
            feed_forward_network = FeedForwardNetwork(d_model, hidden_size, dropout)
            decoder = DecoderBlock(d_model, masked_attention_layer, cross_attention_layer, feed_forward_network, dropout)
            decoder_layers.append(decoder)
        
        self.decoder_layers = nn.ModuleList(decoder_layers)

    def forward(self, x, tgt_mask, enc_output, src_mask) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): decoder input of shape (batch_size, tgt_seq_len, d_model).
            tgt_mask (torch.Tensor): target mask.
            enc_output (torch.Tensor): encoder output of shape (batch_size, src_seq_len, d_model).
            src_mask (torch.Tensor): source mask.

        Returns:
            x (torch.Tensor): Resultant matrix of shape (batch_size, seq_len, d_model).
        """
        for layer in self.decoder_layers:
            x = layer(x, tgt_mask, enc_output, src_mask)
        
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    Projects the output of decoder to vocabulory size vector.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Args:
            vocab_size (int): Number of words in the vocabulory.
            d_model (int): The dimension of each embedding vector.
        """
        super().__init__()
        self.projection_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Matrix of shape (batch_size, d_model).

        Returns:
            x (torch.Tensor): Resultant matrix of shape (batch_size, vocab_size).
        """
        x = self.projection_layer(x)
        return x 

# Decoding Strategy 1: Greedy Decoding
def greedy_decode(model, src, src_mask, tgt_tokenizer, seq_len, device, verbose=False):
    """
    Performs greedy decoding

    Args:
        model (nn.Module): transformer model 
        src (torch.Tensor): source input of shape (1, src_seq_len), containing token IDs.
        src_mask (torch.Tensor): source mask of shape (1, 1, 1, src_seq_len).
        tgt_tokenizer: target tokenizer
        seq_len (int): length of the sequence
        device (torch.device): CPU / CUDA
    """
    sos_idx = tgt_tokenizer.token_to_id('[SOS]')
    eos_idx = tgt_tokenizer.token_to_id('[EOS]')

    enc_output = model.encode(src, src_mask)

    dec_input = torch.empty(1, 1).fill_(sos_idx).type_as(src).to(device)
    while True:
        if dec_input.size(1) == seq_len:
            break

        tgt_mask = causal_mask(dec_input.size(1)).type_as(src_mask).to(device)

        out = model.decode(dec_input, tgt_mask, enc_output, src_mask)

        prob = model.project(out[:, -1])

        if (verbose):
            print(prob)

        _, next_word = torch.max(prob, dim=1)
        dec_input = torch.cat(
            [dec_input, torch.empty(1, 1).type_as(src).fill_(next_word.item()).to(device)], dim=1
        )

        if (verbose):
            print(next_word)

        if next_word == eos_idx:
            break

    return dec_input.squeeze(0)

# Decoding Strategy 2: Top K Decoding
def top_k_decode(model, src, src_mask, tgt_tokenizer, seq_len, device, k=5, verbose=False):
    """
    Performs top k decoding

    Args:
        model (nn.Module): transformer model 
        src (torch.Tensor): source input of shape (1, src_seq_len), containing token IDs.
        src_mask (torch.Tensor): source mask of shape (1, 1, 1, src_seq_len).
        tgt_tokenizer: target tokenizer
        seq_len (int): length of the sequence
        device (torch.device): CPU / CUDA
        k (int): number of candidates to be considered
    """
    sos_idx = tgt_tokenizer.token_to_id('[SOS]')
    eos_idx = tgt_tokenizer.token_to_id('[EOS]')

    enc_output = model.encode(src, src_mask)

    dec_input = torch.empty(1, 1).fill_(sos_idx).type_as(src).to(device)
    while True:
        if dec_input.size(1) == seq_len:
            break

        tgt_mask = causal_mask(dec_input.size(1)).type_as(src_mask).to(device)

        out = model.decode(dec_input, tgt_mask, enc_output, src_mask)

        prob = model.project(out[:, -1]).squeeze(0)

        if (verbose):
            print(prob)

        topk_probs, topk_indices = torch.topk(prob, k)
        topk_probs = topk_probs.cpu().numpy()
        topk_indices = topk_indices.cpu().numpy()

        # Normalizing the probabilities
        sum = topk_probs.sum()
        normalized_probs = topk_probs / sum

        next_word = np.random.choice(topk_indices, 1, p=normalized_probs)[0]

        dec_input = torch.cat(
            [dec_input, torch.empty(1, 1).type_as(src).fill_(next_word).to(device)], dim=1
        )

        if (verbose):
            print(next_word)

        if next_word == eos_idx:
            break

    return dec_input.squeeze(0)

# Decoding Strategy 3: Beam Search
def beam_search_decode(model, src, src_mask, tgt_tokenizer, seq_len, device, beam_size=3, verbose=False):
    """
    Performs beam search decoding.

    Args:
        model (nn.Module): Transformer model
        src (torch.Tensor): source input of shape (1, src_seq_len), containing token IDs
        src_mask (torch.Tensor): source mask of shape (1, 1, 1, src_seq_len)
        tgt_tokenizer: target tokenizer
        seq_len (int): maximum target sequence length
        device (torch.device): CPU / CUDA
        beam_size (int): number of beams to keep
    """
    sos_idx = tgt_tokenizer.token_to_id('[SOS]')
    eos_idx = tgt_tokenizer.token_to_id('[EOS]')

    # Encode the source
    enc_output = model.encode(src, src_mask)

    # Each beam is (sequence tensor, score)
    beams = [(torch.tensor([[sos_idx]], device=device), 0.0)]

    for i in range(seq_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_idx:  
                # If already ended, keep it as is
                new_beams.append((seq, score))
                continue

            tgt_mask = causal_mask(seq.size(1)).type_as(src_mask).to(device)

            out = model.decode(seq, tgt_mask, enc_output, src_mask)
            prob = F.log_softmax(model.project(out[:, -1]), dim=-1)  # use log probs

            topk_probs, topk_indices = torch.topk(prob, beam_size, dim=-1)

            for k in range(beam_size):
                next_word = topk_indices[0, k].item()
                next_score = score + topk_probs[0, k].item()

                new_seq = torch.cat([seq, torch.tensor([[next_word]], device=device)], dim=1)
                new_beams.append((new_seq, next_score))

        # Keep only best beam_size beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        if verbose:
            print("Step beams:")
            for seq, sc in beams:
                print([tgt_tokenizer.id_to_token(t.item()) for t in seq[0]], "score:", sc)

        # If all beams ended with EOS → stop early
        if all(seq[0, -1].item() == eos_idx for seq, _ in beams):
            break

    # Return the best sequence (highest score)
    best_seq = beams[0][0]
    return best_seq.squeeze(0)
