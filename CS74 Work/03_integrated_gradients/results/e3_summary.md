# E3 results: integrated gradients, keyword share

## Convergence (kept if |delta| <= 5% of the sum of |word scores|)

| Model | Sentences | Kept | Dropped |
|---|---|---|---|
| Original | 213 | 198 | 0 real, 15 pair |
| BERT (clean, seed 42) | 213 | 210 | 0 real, 3 pair |
| RoBERTa (clean, seed 42) | 212 | 159 | 12 real, 41 pair |

## Real sentences: keyword share (target = model's predicted class)

Right/wrong = that model's prediction vs. Claude's label. Share = keyword words' |score| as % of all words.

| Model | Class | Right/wrong | n | Mean share (abs) | Median (abs) | Mean share (positive-only) |
|---|---|---|---|---|---|---|
| Original | feminine | wrong | 16 | 26% | 26% | 15% |
| Original | feminine | right | 17 | 40% | 38% | 48% |
| Original | masculine | wrong | 16 | 24% | 19% | 18% |
| Original | masculine | right | 17 | 53% | 51% | 66% |
| Original | neutral | wrong | 16 | 33% | 31% | 42% |
| Original | neutral | right | 17 | 11% | 13% | 5% |
| Original | all | wrong | 48 | 27% | 26% | 25% |
| Original | all | right | 51 | 34% | 36% | 40% |
| Original | all | all | 99 | 31% | 29% | 32% |
| BERT (clean, seed 42) | feminine | wrong | 18 | 25% | 21% | 9% |
| BERT (clean, seed 42) | feminine | right | 15 | 38% | 39% | 46% |
| BERT (clean, seed 42) | masculine | wrong | 14 | 20% | 16% | 15% |
| BERT (clean, seed 42) | masculine | right | 19 | 49% | 47% | 68% |
| BERT (clean, seed 42) | neutral | wrong | 14 | 27% | 23% | 34% |
| BERT (clean, seed 42) | neutral | right | 19 | 10% | 9% | 1% |
| BERT (clean, seed 42) | all | wrong | 46 | 24% | 19% | 19% |
| BERT (clean, seed 42) | all | right | 53 | 32% | 32% | 38% |
| BERT (clean, seed 42) | all | all | 99 | 28% | 25% | 29% |
| RoBERTa (clean, seed 42) | feminine | wrong | 13 | 16% | 10% | 11% |
| RoBERTa (clean, seed 42) | feminine | right | 16 | 25% | 23% | 27% |
| RoBERTa (clean, seed 42) | masculine | wrong | 12 | 19% | 19% | 9% |
| RoBERTa (clean, seed 42) | masculine | right | 21 | 35% | 38% | 39% |
| RoBERTa (clean, seed 42) | neutral | wrong | 14 | 26% | 23% | 31% |
| RoBERTa (clean, seed 42) | neutral | right | 10 | 10% | 8% | 4% |
| RoBERTa (clean, seed 42) | all | wrong | 39 | 21% | 16% | 17% |
| RoBERTa (clean, seed 42) | all | right | 47 | 27% | 19% | 27% |
| RoBERTa (clean, seed 42) | all | all | 86 | 24% | 18% | 23% |

## Minimal pairs: does the keyword push harder when it's used in a gendered way?

Target = the keyword's own class for both sentences. Context pairs = your labels give one sentence the keyword's class and the other neutral.

| Model | Context pairs | Keyword pushes harder in gendered-use sentence | Mean keyword score: gendered use | neutral use | Mean share: gendered use | neutral use | Model gets both right |
|---|---|---|---|---|---|---|---|
| Original | 24 | 13/24 | 2.68 | 1.80 | 48% | 31% | 11/24 |
| BERT (clean, seed 42) | 33 | 18/33 | 3.27 | 1.75 | 47% | 40% | 11/33 |
| RoBERTa (clean, seed 42) | 14 | 10/14 | 2.10 | 1.37 | 42% | 35% | 4/14 |

By set (list / off_list / mixed):

| Model | Set | Pairs | Pushes harder in gendered use | Mean score gendered | Mean score neutral |
|---|---|---|---|---|---|
| Original | list | 14 | 10/14 | 4.99 | 3.01 |
| Original | off_list | 10 | 3/10 | -0.55 | 0.10 |
| BERT (clean, seed 42) | list | 21 | 14/21 | 4.97 | 2.60 |
| BERT (clean, seed 42) | off_list | 12 | 4/12 | 0.28 | 0.26 |
| RoBERTa (clean, seed 42) | list | 10 | 8/10 | 2.91 | 1.88 |
| RoBERTa (clean, seed 42) | off_list | 4 | 2/4 | 0.07 | 0.08 |

All pairs (any labels): how much the keyword's push changes between the two sentences

| Model | Pairs | Median change in keyword score | Median |keyword score| |
|---|---|---|---|
| Original | 42 | 1.50 | 2.13 |
| BERT (clean, seed 42) | 54 | 1.56 | 1.71 |
| RoBERTa (clean, seed 42) | 22 | 0.31 | 1.36 |
