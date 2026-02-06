import torch
from pathlib import Path
from utils import causal_mask
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer, models, trainers
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from datasets import load_dataset

class EnFiDataset(Dataset):
    """
    Dataset class for English-Finnish translation task 
    """
    def __init__(self, dataset, src_tokenizer, tgt_tokenizer, seq_len):
        """
        Args:
            dataset (list): HuggingFace dataset of translation pairs.
            src_tokenizer (Tokenizer): Tokenizer for source language.
            tgt_tokenizer (Tokenizer): Tokenizer for target language.
            src_lang (str): Source language code ('en').
            tgt_lang (str): Target language code ('fi').
            seq_len (int): Maximum sequence length.
        """
        super().__init__()
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tgt_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tgt_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tgt_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
    
    def __len__(self):
        """
        Returns:
            int: Number of samples in dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample.

        Returns:
            dict: Dictionary with enc_input, dec_input, label, text, and masks.
        """
        src_tgt_pair = self.dataset[index]

        src_text = src_tgt_pair["translation"]["en"]    # English is the source
        tgt_text = src_tgt_pair["translation"]["fi"]    # Finnish is the target

        # The tokenizer will convert tokens to ids
        enc_input_tokens = self.src_tokenizer.encode(src_text).ids
        dec_input_tokens = self.tgt_tokenizer.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2    # Both [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1    # Only [SOS], as [EOS] will be part of label
        
        if (enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0):
            print(f"sequence length {self.seq_len} is too small for {len(enc_input_tokens)} or {len(dec_input_tokens)}.")
        
        enc_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(enc_input_tokens, dtype=torch.int64), 
                self.eos_token,
                torch.tensor(enc_num_padding_tokens*[self.pad_token], dtype=torch.int64)
            ],
            dim = 0
        )
        
        # For example, [SOS] I am doing great
        dec_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                torch.tensor(dec_num_padding_tokens*[self.pad_token], dtype=torch.int64)
            ],
            dim = 0
        )

        # For example, I am doing great [EOS]
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                self.eos_token,
                torch.tensor(dec_num_padding_tokens*[self.pad_token], dtype=torch.int64)
            ],
            dim = 0
        )        

        # Encoder mask is used to mask all the pad tokens
        encoder_mask = (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()    # The shape is (1, 1, 1, seq_len)

        # Decoder mask is used to mask all pad tokens and the tokens that come after the word that has to be predicted
        decoder_mask = (dec_input != self.pad_token).unsqueeze(0).int() & causal_mask(dec_input.size(0))

        return {
            "enc_input": enc_input,
            "dec_input": dec_input,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
            "src_mask": encoder_mask,
            "tgt_mask": decoder_mask
        }

def get_all_sentences(dataset, lang):
    """
    Extracts all sentences from the dataset for the specified language

    Args:
        dataset (List[Dict]): List of data samples, each containing translations.
        lang (str): Language code to extract sentences from (e.g., "en", "fi").

    Returns:
        List[str]: List of sentences in the specified language.
    """
    sentences = []
    for entry in dataset:
        # This keeps the original text as it is
        # sentences.append(entry["translation"][lang])    

        # This line removes all the punctuations
        # We do this because the model is only giving punctuations as output
        sentence = entry["translation"][lang]
        clean_sentence = "".join(ch for ch in sentence if ch.isalnum() or ch.isspace()) 
        sentences.append(clean_sentence)
    
    # To print the max sequence length in dataset for lang
    max_len = 0
    for sentence in sentences:
        if(max_len < len(sentence.split())):
            max_len = len(sentence.split())
    print(f"The maximum sequence length in {lang} lang is {max_len}")
    
    return sentences

def get_or_build_tokenizer(path, dataset, lang, tokenizer_type = "BPE"):
    """
    Loads a tokenizer from disk if available, otherwise trains a new one on the dataset and saves it to the given path.

    Args:
        path (str or Path): Path to load or save the tokenizer JSON file.
        dataset (List[Dict]): Dataset used to train tokenizer if needed.
        lang (str): Language to extract and tokenize (e.g., "en", "fi").

    Returns:
        Tokenizer: A trained or loaded tokenizer for the given language.
    """
    path = Path(path)

    if path.exists():
        tokenizer = Tokenizer.from_file(str(path))
        return tokenizer
    else:
        if (tokenizer_type == "BPE"):
            # Use BPE model instead of WordLevel
            tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()

            trainer = trainers.BpeTrainer(
                special_tokens=["[UNK]", "[PAD]", "[EOS]", "[SOS]"],
                vocab_size=32000,   # adjust depending on dataset size
                min_frequency=2
            )

            tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
            tokenizer.save(str(path))
            return tokenizer
        
        # The word level tokenizer leads to the greedy decoding strategy only pick UNK as token
        if (tokenizer_type == "word_level"):
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))    # The tokenizer will assign UNK to each token it does not recognize
            tokenizer.pre_tokenizer = Whitespace()    # The tokenizer will split on whitespace
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[EOS]", "[SOS]"], min_frequency = 2)
            tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
            tokenizer.save(str(path))
            return tokenizer

def create_dataset(data_flag: bool, src_lang: str, tgt_lang: str, seq_len: int, batch_size: int, src_tokenizer_path: str, tgt_tokenizer_path: str, tokenizer_type: str,  src_data_path: str = None, tgt_data_path: str = None):
    """
    Creates training, validation, and test datasets from English-Finnish text files.
    Also builds or loads tokenizers and returns dataloaders.

    Args:
        src_path (str): path to the source (English) text file.
        tgt_path (str): path to the target (Finnish) text file.
        src_tokenizer_path (str): path to the source tokenizer file.
        tgt_tokenizer_path (str): path to the target tokenizer file.
        seq_len (int): sequence length for input/output sentences.
        batch_size (int): batch size for the training dataloader.
        tokenizer_type (str): BPE / word level.

    Returns:
        train_dataloader (DataLoader): dataloader for training data.
        val_dataloader (DataLoader):tdataloader for validation data.
        test_dataloader (DataLoadet): dataloader for test data.
        src_tokenizer (Tokenizer): trained or loaded tokenizer for source language.
        tgt_tokenizer (Tokenizer): trained or loaded tokenizer for target language.
    """
     
    if (data_flag == True):
        with open(src_data_path, encoding="utf-8") as f:
            src_text = [line.strip() for line in f]
        with open(tgt_data_path, encoding="utf-8") as f:
            tgt_text = [line.strip() for line in f]
            
        dataset = [{"translation":{"en": en, "fi": fi}} for en, fi in zip(src_text, tgt_text)]
        dataset = HFDataset.from_list(dataset)
    
    else:
        dataset = load_dataset("Helsinki-NLP/opus_books", f"{src_lang}-{tgt_lang}")

    # Building tokenizer for en and fi
    src_tokenizer = get_or_build_tokenizer(src_tokenizer_path, dataset, "src_lang", tokenizer_type)
    tgt_tokenizer = get_or_build_tokenizer(tgt_tokenizer_path, dataset, "tgt_lang", tokenizer_type)

    # Splitting the data into train/test/val
    ds = dataset.train_test_split(test_size=0.2)
    train_dataset = ds["train"]    # Train is 80% of dataset

    ds = ds["test"].train_test_split(test_size=0.5)  
    val_dataset = ds["train"]    # Val is 10% of dataset
    test_dataset = ds["test"]    # Test is 10% of dataset

    print(f"length of dataset: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")

    train_dataset = EnFiDataset(train_dataset, src_tokenizer, tgt_tokenizer, seq_len)
    val_dataset = EnFiDataset(val_dataset, src_tokenizer, tgt_tokenizer, seq_len)
    test_dataset = EnFiDataset(test_dataset, src_tokenizer, tgt_tokenizer, seq_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    return train_dataloader, val_dataloader, test_dataloader, src_tokenizer, tgt_tokenizer