# Transformer from Scratch

## Table of Contents

- [Transformers-from-Scratch](#Transformers-from-Scratch)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Model Configuration](#model-configuration)
  - [Results](#results)
  - [Documentation](#documentation)
  - [File Structure](#file-structure)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Training](#training)
    - [Testing](#testing)
    - [Pretrained Model Checkpoints](#pretrained-model-checkpoints)
  - [Contributors](#contributors)
  - [References](#references)
  

## About

This project implements a Transformer model from scratch for English-to-Finnish translation. We implement 3 positional embedding techniques (sinusoidal, RoPE, and relative bias), 3 decoding strategies (greedy decoding, top-k sampling, and beam search) and 2 tokenizer (wordlevel and BPE). This project was done as part of assignment of the Advanced NLP course at IIITH.

**Note:** An implementation for vision transformer can be found in `vision-transformer` branch.

## Model Configuration

| **Category**            | **Parameter**               | **Value**                                     |
|-------------------------|-----------------------------|-----------------------------------------------|
| **Model Architecture**   | Sequence Length (`seq_len`) | 500                                           |
|                         | Model Dimension (`d_model`) | 128                                           |
|                         | Number of Encoders          | 3                                             |
|                         | Number of Decoders          | 3                                             |
|                         | Number of Attention Heads   | 4                                             |
|                         | Hidden Size (`hidden_size`) | 256                                           |
|                         | Dropout Rate (`dropout`)    | 0.1                                           |
| **Tokenizer**            | Tokenizer Type              | BPE (Byte Pair Encoding)  / Word Level                   |
| **Positional Encoding**  | Encoding Method (`pe_method`)| ROPE (Relative Positional Encoding) / Relative Bias / Sinusoidal         |
| **Training Hyperparameters** | Batch Size (`batch_size`)  | 32                                            |
|                         | Validation Batches (`val_num_batches`) | 2                                   |
|                         | Alpha (`alpha`)             | 0.05                                          |
|                         | Number of Epochs (`num_epochs`) | 9                                          |
| **Data Paths**           | English Data Path (`eng_path`) | `./data/EUbookshop.en`                     |
| **Decoding**             | Decoding Strategy           | Greedy Decoding / Top-K Sampling, Beam Search                         |

Detailed model configuration is available in `config.py`.

## Results

The results obtained from all the configurations are very trivial. One of the reasons is that the model is too small and the training has been only for 10 epochs. We use BLEU score as the metric.

|                    | **Sinusoidal** | **Rope** | **Relative Bias** |
|--------------------|------------|------|---------------|
| **Greedy Decoding** | NA       | 0 | 0          |
| **Top-k Sampling**  | NA       | 3.168951015957753e-232 | 8.358106388483693e-236          |
| **Beam Search**     | NA       | 0 | 0          |


For methods like Greedy Decoding, the model keeps outputing the same token repeatedly. The token is sometimes `[UNK]` token or random punctuations. This is one issue we need to fix for greedy method. The training logs can be found in `./assets/logs`.

However, one key observation is that RoPE tends to converge faster than other positional encodings. This is evident from the training loss plots.

## Documentation

## File Structure
```
👨‍💻
 ┣ 📂assets 
 ┣ 📂data  
 ┃ ┣ 📄EUbookshop.en 
 ┃ ┣ 📄EUbookshop.fi 
 ┣ 📂documentation  
 ┃ ┣ 📄README.md  
 ┣ 📄config.py
 ┣ 📄dataset.py
 ┣ 📄decoder.py
 ┣ 📄encoder.py
 ┣ 📄train.py
 ┣ 📄test.py
 ┣ 📄utils.py
 ┣ 📄requirements.txt
 ┣ 📄report.pdf
 ┣ 📄README.md
``` 

## Getting Started

### Installation

1) Create a virtual environment

`python3 -m venv tfs`

2) Activate the virtual environment

`source tfs/bin/activate`  (For Linux)

`.\tfs\Scripts\activate`  (For Windows)

3) Install the required dependencies

`pip install -r requirements.txt`

### Training

To configure training parameters (e.g. batch size, learning rate, model type), edit the `config.py` file.

Run training using:

`python train.py`

This will:
1) Start training from scratch or resume from a checkpoint (if specified in config.py)
2) Save model checkpoints after every epoch to the `./checkpoints/` directory with filename as `./checkpoints/{config.pe_method}_model_{epoch}`.
3) Run validation after each epoch and provide BLEU score. Additionally, it will print few examples for each decoding strategy.
4) Plot the training loss after completion of all epochs

Note: The current configuration requires approximately 22 mins per epoch. The google colab file for training / testing the model is available <a href="https://colab.research.google.com/drive/1VCITyBOZafxaeiieBl1l-Ydf10cuTrSd">here</a>.

### Testing

To evaluate a trained model:

`python test.py`

This will:
1) Run model on test dataset and provide BLEU score. 
2) Print a few examples from test set showing source, target and predicted output.


## References
* <a href="https://www.youtube.com/watch?v=ISNdQcPhsts&t=2729s">YouTube Video</a> by Umar Jamil on developing transformers from scratch.
* <a href="https://arxiv.org/abs/1706.03762">Link</a> to ```Attention is all you need``` paper explaining transformer architecture
 

