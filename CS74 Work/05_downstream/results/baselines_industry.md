# E5: keyword rules on the 10,000 postings

% of postings with at least one masculine sentence (CS72's metric). Original BERT (argmax), for comparison: finance 77.0, tech 67.4, education 63.9, healthcare 56.6.

| Labeler | Masc. sentences | Finance | Tech | Education | Healthcare | Gap | Ranking | Holds? |
|---|---|---|---|---|---|---|---|---|
| List rule | 38,322 | 93.4% | 92.6% | 91.5% | 87.2% | 3.7 pts | finance > tech > education > healthcare | yes |
| Learned rule (clean train) | 20,516 | 86.2% | 80.3% | 78.7% | 74.3% | 6.7 pts | finance > tech > education > healthcare | yes |
| Learned rule (CS72 train) | 21,964 | 86.7% | 82.4% | 79.7% | 74.7% | 7.3 pts | finance > tech > education > healthcare | yes |
