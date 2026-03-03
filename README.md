# Natural_Language_Processing_Assignment_6_Mingxuan-Li
AIG230 – Assignment 6
1. Overview
This assignment compares two approaches to language modeling using the Brown corpus (news category): a statistical trigram language model with Laplace smoothing, and a neural language model using a PyTorch RNN. Both models use identical preprocessing, vocabulary construction, and data splits. Performance is evaluated using perplexity and qualitative text generation analysis.
2. Dataset and Preprocessing
Source: Brown corpus (news category)
Raw sentences: 4623
Sentences after preprocessing: 4611
Preprocessing steps:
- Lowercasing tokens
- Removing punctuation-only tokens
- Keeping stopwords
- Adding <bos> and <eos> tokens
3. Data Split
Train: 3698 sentences | 77,918 tokens
Validation: 462 sentences | 10,264 tokens
Test: 463 sentences | 9,656 tokens
4. Vocabulary
Built from training data only with min_freq = 2
Final vocabulary size: 5,353 tokens
Rare words mapped to <unk>
Part A – Trigram Language Model
Model: Trigram (n=3) with Laplace (Add-1) smoothing
Validation Perplexity: 553.16
Test Perplexity: 531.97
The trigram model captures local word dependencies effectively but lacks long-range contextual modeling. Smoothing prevents zero probabilities for unseen n-grams, ensuring finite perplexity.
Part B – Neural RNN Language Model
Embedding dimension: 128
Hidden dimension: 256
Number of layers: 1
Sequence length: 30
Total parameters: 2,159,721
Training Results
Epoch 1: Train Loss 3.6107 | Val PPL 431.68
Epoch 2: Train Loss 1.2977 | Val PPL 1730.75
Epoch 3: Train Loss 0.6994 | Val PPL 4389.29
Epoch 4: Train Loss 0.5123 | Val PPL 8419.99
Epoch 5: Train Loss 0.4329 | Val PPL 12981.13
Final Test Perplexity: 12166.41
Although the training loss decreases steadily, validation perplexity increases dramatically, indicating severe overfitting. The RNN model memorizes training sequences but fails to generalize.
Conclusion
The trigram model generalizes more robustly with lower perplexity (~532) compared to the RNN (~12166). Despite neural models being theoretically more powerful, insufficient regularization and limited data led to catastrophic overfitting in the RNN. This experiment highlights the importance of model capacity control, regularization, and proper evaluation using perplexity.
