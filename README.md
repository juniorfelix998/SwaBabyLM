# SwaLM2: Curriculum Learning vs. Random Order Learning for Swahili

This repository contains a comparative study of curriculum-based and random order learning approaches for training language models on Swahili text data. The project implements small-scale GPT-2 models ("GPT-Wee") trained on carefully organized data to compare the efficacy of these two learning strategies.

## Overview

The study explores whether a language model trained with a curriculum (progressively increasing text complexity) outperforms one trained with randomly ordered data on grammatical knowledge acquisition in Swahili. We specifically measure this through a custom benchmark called SwaLiMP (Swahili Linguistic Minimal Pairs).

## Features

- Custom Swahili BPE tokenizer
- Small-scale GPT-2 model implementation for Swahili
- Curriculum learning implementation with text ordered by complexity
- Random order learning implementation
- SwaLiMP: A custom benchmark for evaluating grammatical knowledge

## Dataset

The training data consists of three main sources:
- Swahili CHILDES corpus (child-directed speech)
- Swahili children's stories
- Swahili Wikipedia data

The curriculum version orders these texts from simplest to most complex, while the random version mixes them without any specific ordering.

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SwaLM2.git
cd SwaLM2

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries

- PyTorch
- Transformers
- Datasets
- Tokenizers
- Pandas
- NumPy
- Matplotlib
- Seaborn
- tqdm
- wandb (for logging)

## Usage

### 1. Prepare your data

Organize your Swahili text files in the following structure:
```
SwaLM2/
├── train/
│   ├── aochildesswahilitrain.txt
│   ├── childrenstoriestrain.txt
│   └── swahili_wikipedia_data.txt
├── dev/
│   ├── aochildesdevswa.txt
│   └── chlidrenstoriesdevswahili.txt
└── ordered_text.txt  # For curriculum learning
```

### 2. Train the tokenizer

```python
tokenizer = create_swahili_tokenizer(
    swahili_data_path,
    tokenizer_dir,
    vocab_size=8000
)
```

### 3. Prepare datasets

```python
# For curriculum learning
curr_datasets = prepare_datasets(
    tokenizer,
    curr_train_files,
    curr_eval_files,
    is_curriculum=True
)

# For random order learning
rand_datasets = prepare_datasets(
    tokenizer,
    rand_train_files,
    rand_eval_files,
    is_curriculum=False
)
```

### 4. Create and train models

```python
# Create models
curr_model = create_model(tokenizer)
rand_model = create_model(tokenizer)

# Train models
curr_trainer = train_model(
    curr_model,
    tokenizer,
    curr_datasets,
    curriculum_output_dir,
    is_curriculum=True
)

rand_trainer = train_model(
    rand_model,
    tokenizer,
    rand_datasets,
    random_output_dir,
    is_curriculum=False
)
```

### 5. Evaluate on SwaLiMP

```python
# Evaluate models
rand_evaluator = SwaLiMP(rand_model, tokenizer)
rand_accuracy, rand_results = rand_evaluator.evaluate()

curr_evaluator = SwaLiMP(curr_model, tokenizer)
curr_accuracy, curr_results = curr_evaluator.evaluate()
```

## Key Components

### Tokenizer

The project uses a custom BPE tokenizer trained specifically on Swahili text data. The tokenizer normalizes text, applies byte-level pre-tokenization, and has a vocabulary size of 8000 tokens.

### Model Architecture

We use a small-scale version of GPT-2 with the following specifications:
- Embedding dimension: 128
- Number of layers: 2
- Number of attention heads: 2
- Context length: 128 tokens

### SwaLiMP Benchmark

SwaLiMP (Swahili Linguistic Minimal Pairs) is a custom benchmark inspired by BLiMP, designed to evaluate grammatical knowledge in language models. It tests various aspects of Swahili grammar:

1. Subject-Verb Agreement
2. Noun Class Agreement
3. Word Order
4. Tense Marking

For each grammatical category, the benchmark provides pairs of sentences where one is grammatical and the other ungrammatical. A model scores correctly if it assigns higher probability to the grammatical sentence.

## Results

Our experiments show that:

- The curriculum learning model achieves better overall grammatical knowledge compared to the random order model
- Performance varies across grammatical categories:
  - Subject-Verb Agreement: [Results]
  - Noun Class Agreement: [Results]
  - Word Order: [Results]
  - Tense Marking: [Results]
- Training loss and evaluation loss curves show different learning dynamics between the two approaches

## Visualizations

The `plots/` directory contains several visualizations:
- Training loss comparison
- Evaluation loss comparison
- SwaLiMP accuracy comparison
- Performance across grammatical categories
- Learning trajectory

## Future Work

- Scale up model size and training data
- Experiment with different curriculum organizations
- Expand the SwaLiMP benchmark with more linguistic phenomena
- Test transfer learning from curriculum-trained models to downstream tasks


## Citation

If you use this code in your research, please cite:

```bibtex
@misc{SwaLM2,
  author = {Felix Owino},
  title = {SwaLM2: Curriculum Learning vs. Random Order Learning for Swahili},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/juniorfelix998/SwaBabyLM}
}
```