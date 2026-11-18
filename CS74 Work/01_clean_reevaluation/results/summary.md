# E1 results

Macro-F1 unless noted. New models: mean ± SD over 3 seeds. Others: one run.
Per-class F1 columns are on all sentences of the set.

## New evaluation set, unseen sentences (HEADLINE: every model on equal footing)

| Model | n | All | Random | Enriched | Flagged-only | F1 masculine | F1 feminine | F1 neutral |
|---|---|---|---|---|---|---|---|---|
| Original (CS72) | 3,633 | 0.798 | 0.773 | 0.803 | 0.775 | 0.681 | 0.761 | 0.953 |
| BERT (clean) | 3,633 | 0.784 ± 0.003 | 0.773 ± 0.006 | 0.785 ± 0.003 | 0.762 ± 0.006 | 0.684 ± 0.011 | 0.717 ± 0.004 | 0.950 ± 0.000 |
| RoBERTa (clean) | 3,633 | 0.777 ± 0.003 | 0.763 ± 0.002 | 0.778 ± 0.004 | 0.752 ± 0.004 | 0.681 ± 0.013 | 0.707 ± 0.020 | 0.943 ± 0.002 |
| List rule | 3,633 | 0.617 | 0.555 | 0.640 | 0.345 | 0.509 | 0.527 | 0.815 |
| Learned rule (clean train) | 3,633 | 0.685 | 0.661 | 0.690 | 0.632 | 0.588 | 0.533 | 0.934 |
| Learned rule (CS72 train) | 3,633 | 0.703 | 0.687 | 0.706 | 0.648 | 0.616 | 0.562 | 0.932 |

## New evaluation set, all sentences

| Model | n | All | Random | Enriched | Flagged-only | F1 masculine | F1 feminine | F1 neutral |
|---|---|---|---|---|---|---|---|---|
| Original (CS72) | 3,721 | 0.797 | 0.774 | 0.801 | 0.774 | 0.679 | 0.758 | 0.953 |
| BERT (clean) | 3,721 | 0.783 ± 0.003 | 0.774 ± 0.006 | 0.783 ± 0.003 | 0.761 ± 0.006 | 0.681 ± 0.010 | 0.717 ± 0.004 | 0.950 ± 0.000 |
| RoBERTa (clean) | 3,721 | 0.776 ± 0.003 | 0.764 ± 0.002 | 0.777 ± 0.004 | 0.752 ± 0.004 | 0.679 ± 0.012 | 0.707 ± 0.020 | 0.944 ± 0.002 |
| List rule | 3,721 | 0.614 | 0.552 | 0.637 | 0.343 | 0.501 | 0.527 | 0.815 |
| Learned rule (clean train) | 3,721 | 0.685 | 0.661 | 0.690 | 0.633 | 0.586 | 0.534 | 0.935 |
| Learned rule (CS72 train) | 3,721 | 0.703 | 0.688 | 0.705 | 0.649 | 0.614 | 0.562 | 0.933 |

## Clean test set, minus sentences also in train (test_b)

| Model | n | All | Random | Enriched | Flagged-only | F1 masculine | F1 feminine | F1 neutral |
|---|---|---|---|---|---|---|---|---|
| BERT (clean) | 2,733 | 0.817 ± 0.003 | 0.807 ± 0.007 | 0.816 ± 0.002 | 0.796 ± 0.001 | 0.758 ± 0.006 | 0.736 ± 0.008 | 0.959 ± 0.002 |
| RoBERTa (clean) | 2,733 | 0.803 ± 0.005 | 0.804 ± 0.013 | 0.798 ± 0.006 | 0.784 ± 0.008 | 0.736 ± 0.010 | 0.718 ± 0.006 | 0.954 ± 0.001 |
| List rule | 2,733 | 0.617 | 0.601 | 0.619 | 0.341 | 0.517 | 0.508 | 0.826 |
| Learned rule (clean train) | 2,733 | 0.698 | 0.661 | 0.708 | 0.650 | 0.617 | 0.531 | 0.945 |

## Clean test set (test_a)

| Model | n | All | Random | Enriched | Flagged-only | F1 masculine | F1 feminine | F1 neutral |
|---|---|---|---|---|---|---|---|---|
| BERT (clean) | 2,767 | 0.818 ± 0.003 | 0.808 ± 0.007 | 0.817 ± 0.002 | 0.797 ± 0.001 | 0.758 ± 0.006 | 0.737 ± 0.008 | 0.959 ± 0.002 |
| RoBERTa (clean) | 2,767 | 0.803 ± 0.005 | 0.804 ± 0.013 | 0.799 ± 0.006 | 0.785 ± 0.008 | 0.736 ± 0.010 | 0.719 ± 0.006 | 0.954 ± 0.001 |
| List rule | 2,767 | 0.616 | 0.601 | 0.618 | 0.340 | 0.513 | 0.507 | 0.827 |
| Learned rule (clean train) | 2,767 | 0.697 | 0.661 | 0.707 | 0.650 | 0.616 | 0.530 | 0.945 |

## CS72's own test set, as published

| Model | n | All | Random | Enriched | Flagged-only | F1 masculine | F1 feminine | F1 neutral |
|---|---|---|---|---|---|---|---|---|
| Original (CS72) | 3,716 | 0.853 | 0.826 | 0.861 | 0.827 | 0.782 | 0.812 | 0.964 |
| List rule | 3,716 | 0.681 | 0.638 | 0.698 | 0.398 | 0.578 | 0.617 | 0.849 |
| Learned rule (CS72 train) | 3,716 | 0.769 | 0.764 | 0.768 | 0.710 | 0.694 | 0.670 | 0.942 |

## CS72's own test set, without the 24 unlabeled sentences

| Model | n | All | Random | Enriched | Flagged-only | F1 masculine | F1 feminine | F1 neutral |
|---|---|---|---|---|---|---|---|---|
| Original (CS72) | 3,692 | 0.858 | 0.826 | 0.868 | 0.833 | 0.790 | 0.817 | 0.966 |
| List rule | 3,692 | 0.687 | 0.638 | 0.706 | 0.403 | 0.586 | 0.622 | 0.853 |
| Learned rule (CS72 train) | 3,692 | 0.773 | 0.764 | 0.774 | 0.714 | 0.701 | 0.675 | 0.944 |

## Sentences longer than 128 tokens (cut off)

- BERT, test_a: 94 of 2767
- BERT, test_b: 94 of 2733
- BERT, new_eval: 125 of 3721
- RoBERTa, test_a: 95 of 2767
- RoBERTa, test_b: 95 of 2733
- RoBERTa, new_eval: 125 of 3721
