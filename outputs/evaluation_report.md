# Evaluation report

Generated 2026-10-26 22:47:03.

All models were evaluated on the same held-out validation documents.

| Model | Val loss | Val acc | Val top-5 | Perplexity | Train loss | Train acc | Epochs (best) | Time (min) | Params |
|---|---|---|---|---|---|---|---|---|---|
| RNN | 4.7615 | 21.95% | 41.31% | 116.9 | 4.5232 | 22.51% | 10 (10) | 18.3 | 2,055,375 |
| LSTM | 4.7421 | 22.66% | 42.11% | 114.7 | 4.4448 | 23.21% | 10 (10) | 32.9 | 2,141,775 |
| GRU | 4.6234 | 23.14% | 43.03% | 101.8 | 4.3320 | 24.03% | 10 (10) | 21.7 | 2,113,359 |

Unigram baseline (always guess most frequent words): top-1 7.76%, top-5 18.47%, perplexity 675.4.

Validation accuracy measures exact next-word agreement with held-out documents. It is a classification metric, not a guarantee of fluent or correct language.

Training accuracy is measured with dropout active, so it is not directly comparable to validation accuracy.
