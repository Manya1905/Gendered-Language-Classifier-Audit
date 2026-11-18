# E5 results: does the industry ranking depend on the labeler?

% of postings with at least one masculine sentence (CS72's metric), 2,500 postings per industry. Gap = (finance + tech)/2 − (education + healthcare)/2. Holds = finance and tech both above education and healthcare. CS72's ranking: finance > tech > education > healthcare.

## 1. Main result (retrained models: mean ± SD over 3 seeds)

| Labeler | Finance | Tech | Education | Healthcare | Gap | Same ranking as CS72 | Holds |
|---|---|---|---|---|---|---|---|
| Original (CS72) | 77.0% | 67.4% | 63.9% | 56.6% | 12.0 pts | 1/1 | 1/1 |
| BERT (clean) | 76.9 ± 2.2% | 66.2 ± 1.9% | 61.8 ± 1.8% | 52.6 ± 0.8% | 14.3 ± 2.2 pts | 3/3 | 3/3 |
| RoBERTa (clean) | 80.3 ± 0.6% | 70.8 ± 1.8% | 63.8 ± 3.3% | 55.4 ± 1.5% | 15.9 ± 2.0 pts | 3/3 | 3/3 |
| List rule | 93.4% | 92.6% | 91.5% | 87.2% | 3.7 pts | 1/1 | 1/1 |
| Learned rule (clean train) | 86.2% | 80.3% | 78.7% | 74.3% | 6.7 pts | 1/1 | 1/1 |
| Learned rule (CS72 train) | 86.7% | 82.4% | 79.7% | 74.7% | 7.3 pts | 1/1 | 1/1 |

## 2. Every run, with bootstrap

Bootstrap: 1,000 resamples of postings within each industry. "% same ranking" / "% holds" = share of resamples where it still holds.

| Run | Masc. sentences | Gap | Gap 95% CI | Ranking | % same ranking | % holds |
|---|---|---|---|---|---|---|
| Original (CS72) | 13,954 | 12.0 | [10.2, 13.8] | finance > tech > education > healthcare | 100% | 100% |
| BERT (clean) (seed 42) | 12,824 | 12.8 | [11.0, 14.8] | finance > tech > education > healthcare | 92% | 92% |
| BERT (clean) (seed 43) | 13,575 | 16.9 | [15.1, 18.8] | finance > tech > education > healthcare | 100% | 100% |
| BERT (clean) (seed 44) | 12,882 | 13.3 | [11.4, 15.1] | finance > tech > education > healthcare | 99% | 99% |
| RoBERTa (clean) (seed 42) | 14,178 | 15.4 | [13.6, 17.2] | finance > tech > education > healthcare | 100% | 100% |
| RoBERTa (clean) (seed 43) | 14,718 | 18.1 | [16.4, 19.9] | finance > tech > education > healthcare | 100% | 100% |
| RoBERTa (clean) (seed 44) | 16,246 | 14.3 | [12.6, 16.2] | finance > tech > education > healthcare | 100% | 100% |
| List rule | 38,322 | 3.7 | [2.5, 4.8] | finance > tech > education > healthcare | 80% | 92% |
| Learned rule (clean train) | 20,516 | 6.7 | [5.2, 8.3] | finance > tech > education > healthcare | 92% | 92% |
| Learned rule (CS72 train) | 21,964 | 7.3 | [5.8, 8.9] | finance > tech > education > healthcare | 99% | 99% |

## 3. Per-sentence masculine rate (% of sentences)

Not affected by posting length.

| Run | Finance | Tech | Education | Healthcare | Holds |
|---|---|---|---|---|---|
| Original (CS72) | 8.06% | 7.58% | 5.63% | 5.24% | yes |
| BERT (clean) (seed 42) | 7.41% | 7.12% | 5.25% | 4.58% | yes |
| BERT (clean) (seed 43) | 8.17% | 7.58% | 5.21% | 4.93% | yes |
| BERT (clean) (seed 44) | 7.73% | 7.05% | 5.05% | 4.70% | yes |
| RoBERTa (clean) (seed 42) | 8.71% | 7.71% | 5.49% | 5.11% | yes |
| RoBERTa (clean) (seed 43) | 9.09% | 8.30% | 5.48% | 5.25% | yes |
| RoBERTa (clean) (seed 44) | 9.71% | 9.21% | 6.15% | 5.94% | yes |
| List rule | 20.74% | 20.37% | 16.13% | 15.26% | yes |
| Learned rule (clean train) | 10.85% | 10.74% | 8.73% | 8.45% | yes |
| Learned rule (CS72 train) | 11.40% | 11.70% | 9.51% | 8.86% | yes |

## 4. One posting per company (3,765 postings)

| Run | Finance | Tech | Education | Healthcare | Gap | Ranking | Holds |
|---|---|---|---|---|---|---|---|
| Original (CS72) | 75.4% | 67.9% | 69.2% | 59.5% | 7.3 | finance > education > tech > healthcare | NO |
| BERT (clean) (seed 42) | 72.5% | 66.3% | 68.1% | 54.8% | 7.9 | finance > education > tech > healthcare | NO |
| BERT (clean) (seed 43) | 74.8% | 68.7% | 70.2% | 57.7% | 7.8 | finance > education > tech > healthcare | NO |
| BERT (clean) (seed 44) | 71.4% | 64.3% | 65.1% | 55.1% | 7.8 | finance > education > tech > healthcare | NO |
| RoBERTa (clean) (seed 42) | 76.3% | 67.6% | 68.0% | 57.7% | 9.1 | finance > education > tech > healthcare | NO |
| RoBERTa (clean) (seed 43) | 75.8% | 69.9% | 70.6% | 59.4% | 7.8 | finance > education > tech > healthcare | NO |
| RoBERTa (clean) (seed 44) | 76.8% | 71.8% | 70.5% | 60.4% | 8.9 | finance > tech > education > healthcare | yes |
| List rule | 93.7% | 92.5% | 92.1% | 89.2% | 2.4 | finance > tech > education > healthcare | yes |
| Learned rule (clean train) | 83.1% | 78.8% | 81.9% | 77.5% | 1.3 | finance > education > tech > healthcare | NO |
| Learned rule (CS72 train) | 83.1% | 81.9% | 83.7% | 78.1% | 1.6 | education > finance > tech > healthcare | NO |
