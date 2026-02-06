# This script contains all the configurations required for model
class Config:
    # Data
    data_flag: 0    # 0 will use hugging face dataset, and 1 will use custom dataset
    src_lang = "fi"
    tgt_lang = "en"

    # Model Architecture
    seq_len = 500
    d_model = 512
    num_encoders = 6
    num_decoders = 6
    num_heads = 8
    hidden_size = 1024
    dropout = 0.1

    # Tokenizer
    tokenizer_type = "BPE"    # BPE / word_level

    # Positional Encoding
    pe_method = "rope"  # sinusoidal / relative_bias / rope

    # Training Hyperparameters 
    batch_size = 32
    val_num_batches = 2
    alpha = 0.05  
    num_epochs = 9

    # Data Paths
    eng_path = "./data/EUbookshop.en"
    fin_path = "./data/EUbookshop.fi"
    eng_tokenizer = "./tokenizer_en.json"
    fin_tokenizer = "./tokenizer_fi.json"

    # Decoding
    decoding_strategy = ["greedy", "top_k", "beam"]  # greedy / beam / topk

    # Experiment & Logging
    experiment_name = "exp0"
    preload = False    # For preloading a model during training, this has to be True  

    # Best Model
    model_path = "./checkpoints/rope_model_8"
