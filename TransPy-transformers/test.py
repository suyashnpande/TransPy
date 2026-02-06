import nltk
import torch
from config import Config
from train import Transformer
from dataset import create_dataset
from decoder import greedy_decode, top_k_decode, beam_search_decode

def test_model(config, num_examples = 2, num_batch = 2):
    """
    Trains the Transformer model using the specified configuration.

    Args:
        config (Config): Configuration object containing training parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer = create_dataset(config.eng_path, config.fin_path, config.eng_tokenizer, config.fin_tokenizer, config.seq_len, config.batch_size, config.tokenizer_type)
    model = Transformer(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)

    model_path = config.model_path
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loading preloaded model from {model_path}")

    model.eval()
    print(f"Model loaded")

    console_width = 80
    source_texts = []
    expected = []
    predicted_greedy = []
    predicted_topk = []
    predicted_beam = []

    with torch.no_grad():
        example_count = 0
        batch_count = 0
        verbose = True
        for batch in test_loader:
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
                    model_out_beam = beam_search_decode(model, src, src_mask, tgt_tokenizer, config.seq_len, device, 10)

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

if __name__ == '__main__':
    config = Config()
    test_model(config, num_examples = 1, num_batch = 10)