import os
import torch
import nltk
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import Config
from dataset import create_dataset
from utils import EmbeddingLayer, SinusoidalPositionalEncoding, RotaryPositionalEncoding, RelativePositionBias, MultiHeadAttentionLayer
from encoder import Encoder
from decoder import Decoder, ProjectionLayer, greedy_decode, top_k_decode, beam_search_decode

class Transformer(nn.Module):
    """
    Full Transformer architecture with embedding, encoder, decoder, and projection layer.
    """
    def __init__(self, config, src_vocab_size, tgt_vocab_size) -> None:
        """
        Args:
            config (Config): Configuration object.
            src_vocab_size (int): Size of source language vocabulary.
            tgt_vocab_size (int): Size of target language vocabulary.
        """
        super().__init__()
        self.config = config
        self.src_embedding_layer = EmbeddingLayer(src_vocab_size, config.d_model)
        self.tgt_embedding_layer = EmbeddingLayer(tgt_vocab_size, config.d_model)

        if (config.pe_method == "sinusoidal"):
            self.src_positional_encodings = SinusoidalPositionalEncoding(config.seq_len, config.d_model)
            self.tgt_positional_encodings = SinusoidalPositionalEncoding(config.seq_len, config.d_model)

            self.encoder = Encoder(config.num_encoders, config.d_model, config.num_heads, config.hidden_size, config.seq_len, config.pe_method, config.dropout)
            self.decoder = Decoder(config.num_decoders, config.d_model, config.num_heads, config.hidden_size, config.seq_len, config.pe_method, config.dropout)
        
        if (config.pe_method == "rope"):
            self.src_positional_encodings = RotaryPositionalEncoding(config.seq_len, config.d_model, config.num_heads)
            self.tgt_positional_encodings = RotaryPositionalEncoding(config.seq_len, config.d_model, config.num_heads)

            self.encoder = Encoder(config.num_encoders, config.d_model, config.num_heads, config.hidden_size, config.seq_len, config.pe_method, config.dropout, self.src_positional_encodings)
            self.decoder = Decoder(config.num_decoders, config.d_model, config.num_heads, config.hidden_size, config.seq_len, config.pe_method, config.dropout, self.tgt_positional_encodings)
        
        if (config.pe_method == "relative_bias"):
            self.src_positional_encodings = RelativePositionBias(config.seq_len, config.num_heads)
            self.tgt_positional_encodings = RelativePositionBias(config.seq_len, config.num_heads)

            self.encoder = Encoder(config.num_encoders, config.d_model, config.num_heads, config.hidden_size, config.seq_len, config.pe_method, config.dropout, self.src_positional_encodings)
            self.decoder = Decoder(config.num_decoders, config.d_model, config.num_heads, config.hidden_size, config.seq_len, config.pe_method, config.dropout, self.tgt_positional_encodings)

        self.projection_layer = ProjectionLayer(config.d_model, tgt_vocab_size)
    
    def encode(self, src_input, src_mask):
        """
        Args:
            src_input (torch.Tensor): Source input tensor of shape (batch_size, seq_len).
            src_mask (torch.Tensor): Mask for source tokens.

        Returns:
            torch.Tensor: Encoded representation of shape (batch_size, seq_len, d_model).
        """
        x = self.src_embedding_layer(src_input)
        if (self.config.pe_method == "sinusoidal"):
            x = self.src_positional_encodings(x)

        x = self.encoder(x, src_mask)
        return x
    
    def decode(self, x, tgt_mask, enc_output, src_mask, verbose = None):
        """
        Args:
            x (torch.Tensor): decoder input of shape (batch_size, tgt_seq_len, d_model).
            tgt_mask (torch.Tensor): target mask.
            enc_output (torch.Tensor): encoder output of shape (batch_size, src_seq_len, d_model).
            src_mask (torch.Tensor): source mask.

        Returns:
            torch.Tensor: Decoder output of shape (batch_size, seq_len, d_model).
        """
        x = self.tgt_embedding_layer(x)
        if (self.config.pe_method == "sinusoidal"):
            x = self.tgt_positional_encodings(x)
        x = self.decoder(x, tgt_mask, enc_output, src_mask)
        return x
    
    def project(self, x):
        """
        Args:
            x (torch.Tensor): Decoder output tensor.

        Returns:
            torch.Tensor: Logits over vocabulary of shape (batch_size, vocab_size).
        """
        return self.projection_layer(x)

def train_model(config):
    """
    Trains the Transformer model using the specified configuration.

    Args:
        config (Config): Configuration object containing training parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    os.makedirs("./checkpoints", exist_ok=True)

    train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer = create_dataset(config.eng_path, config.fin_path, config.eng_tokenizer, config.fin_tokenizer, config.seq_len, config.batch_size, config.tokenizer_type)

    model = Transformer(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)

    # Xavier initialization for all parameters with dim > 1
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.alpha, eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id("[PAD]"))

    # Lists to keep track of losses per epoch
    train_losses = []

    start_epoch = 0
    global_step = 0

    if config.preload is True:
        model_path = config.model_path
        print(f"Loading preloaded model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]

    for epoch in range(start_epoch, config.num_epochs):
        torch.cuda.empty_cache()
        model.train()
        epoch_loss = 0
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")

        for batch in batch_iterator:
            enc_input = batch["enc_input"].to(device)
            dec_input = batch["dec_input"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            encoder_output = model.encode(enc_input, src_mask)
            decoder_output = model.decode(dec_input, tgt_mask, encoder_output, src_mask)
            projection_output = model.project(decoder_output)

            loss = loss_fn(projection_output.view(-1, tgt_tokenizer.get_vocab_size()), labels.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"\nEpoch {epoch+1} completed. Average Training Loss: {avg_loss:.4f}.")

        # Save checkpoint after each epoch
        checkpoint_path = f"checkpoints/{config.pe_method}_model_{epoch}"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, checkpoint_path)

        print(f"Checkpoint saved at checkpoints/{config.pe_method}_model_{epoch}.")

        # Run validation 
        run_validation(model, val_loader, src_tokenizer, tgt_tokenizer, config, device, num_examples=2, num_batch=config.val_num_batches)
        print(f"Validation for epoch {epoch+1} completed.")

    return train_losses

def run_validation(model, val_dataloader, src_tokenizer, tgt_tokenizer, config, device, num_examples=2, num_batch = 10):
    """
    Runs the model on validation set.
    Args:
        model (Tansformer): trained transformer model.
        val_dataloader (Dataloader): validation data.
        tgt_tokenizer (Tokenizer): target tokenizer.
        config (Config): configuration. 
        device: CPU / CUDA.
        num_examples (int): number of examples to be printed after validation run.
    """
    model.eval()
    console_width = 80

    # Debugging the tokens generated by the model
    # sos_idx = src_tokenizer.token_to_id('[SOS]')
    # eos_idx = src_tokenizer.token_to_id('[EOS]')
    # pad_idx = src_tokenizer.token_to_id('[PAD]')
    # print(f"Source side: {sos_idx}, {eos_idx}, {pad_idx}")
    # print(src_tokenizer.id_to_token(0))
    # print("/"*console_width)
    # sos_idx = tgt_tokenizer.token_to_id('[SOS]')
    # eos_idx = tgt_tokenizer.token_to_id('[EOS]')
    # pad_idx = tgt_tokenizer.token_to_id('[PAD]')
    # print(f"Target side: {sos_idx}, {eos_idx}, {pad_idx}")
    # print(tgt_tokenizer.id_to_token(0))
    
    source_texts = []
    expected = []
    predicted_greedy = []
    predicted_topk = []
    predicted_beam = []

    with torch.no_grad():
        example_count = 0
        batch_count = 0
        verbose = True
        for batch in val_dataloader:
            encoder_input = batch["enc_input"].to(device)     # shape: (batch_size, seq_len)
            encoder_mask = batch["src_mask"].to(device)       # shape: (batch_size, 1, 1, seq_len) or similar
            batch_size = encoder_input.size(0)

            # Process each example individually for greedy decoding
            for i in range(batch_size):
                src = encoder_input[i].unsqueeze(0)           # (1, seq_len)
                src_mask = encoder_mask[i].unsqueeze(0)       # (1, 1, 1, seq_len) or (1, 1, seq_len) depending on your mask shape

                if "greedy" in config.decoding_strategy :
                    model_out_greedy = greedy_decode(model, src, src_mask, tgt_tokenizer, config.seq_len, device)
                if "top_k" in config.decoding_strategy:
                    model_out_topk = top_k_decode(model, src, src_mask, tgt_tokenizer, config.seq_len, device, 5)
                if "beam" in config.decoding_strategy:
                    model_out_beam = beam_search_decode(model, src, src_mask, tgt_tokenizer, config.seq_len, device, 3)

                source_text = batch["src_text"][i]
                target_text = batch["tgt_text"][i]

                if "greedy" in config.decoding_strategy :
                    model_out_greedy_text = tgt_tokenizer.decode(model_out_greedy.detach().cpu().numpy())
                    predicted_greedy.append(model_out_greedy_text)
                if "top_k" in config.decoding_strategy:
                    model_out_topk_text = tgt_tokenizer.decode(model_out_topk.detach().cpu().numpy())
                    predicted_topk.append(model_out_topk_text)
                if "beam" in config.decoding_strategy:
                    model_out_beam_text = tgt_tokenizer.decode(model_out_beam.detach().cpu().numpy())
                    predicted_beam.append(model_out_beam_text) 

                source_texts.append(source_text)
                expected.append(target_text)

                # Print only num_examples samples
                if example_count < num_examples:
                    print('-' * console_width)
                    print(f"Source: {source_text}")
                    print(f"Target: {target_text}")
                    if "greedy" in config.decoding_strategy :
                        print(f"Greedy Decoding: {model_out_greedy_text}")
                    if "top_k" in config.decoding_strategy:
                        print(f"Top K Sampling: {model_out_topk_text}")
                    if "beam" in config.decoding_strategy:
                        print(f"Beam Search: {model_out_beam_text}")
                    example_count += 1
                
                if example_count >= num_examples:
                    verbose = False

            batch_count += 1
            if (batch_count >= num_batch):
                break

    # BLEU score
    references = [[ref.split()] for ref in expected]
    if "greedy" in config.decoding_strategy :
        candidates = [pred.split() for pred in predicted_greedy]
        bleu_score = nltk.translate.bleu_score.corpus_bleu(references, candidates)
        print(f"BLEU score using Greedy Decoding: {bleu_score}")
    if "top_k" in config.decoding_strategy:
        candidates = [pred.split() for pred in predicted_topk]
        bleu_score = nltk.translate.bleu_score.corpus_bleu(references, candidates)
        print(f"BLEU score using Top K Sampling: {bleu_score}")
    if "beam" in config.decoding_strategy:
        candidates = [pred.split() for pred in predicted_beam]
        bleu_score = nltk.translate.bleu_score.corpus_bleu(references, candidates)
        print(f"BLEU score using Beam Search: {bleu_score}")

# if __name__ == '__main__':
#     config = Config()
#     train_losses = train_model(config)

#     # Plot training loss
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color="red")
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Loss Over Epochs')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('./training_loss.png')
#     plt.show()