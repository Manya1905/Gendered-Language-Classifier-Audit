# E5 part 2: threshold sweep (original model)

% of postings with at least one masculine sentence (2,500 postings per industry).
Masculine at threshold t = P(masculine) >= t. "Holds" = finance and tech both above education and healthcare.

| Threshold | Masc. sentences | Finance | Tech | Education | Healthcare | Male-dom avg | Female-dom avg | Gap | Ranking | Holds? |
|---|---|---|---|---|---|---|---|---|---|---|
| argmax (CS72) | 13,954 | 77.0% | 67.4% | 63.9% | 56.6% | 72.2% | 60.2% | 12.0 pts | finance > tech > education > healthcare | yes |
| 0.50 | 13,914 | 77.0% | 67.4% | 63.6% | 56.4% | 72.2% | 60.0% | 12.2 pts | finance > tech > education > healthcare | yes |
| 0.55 | 13,725 | 76.6% | 67.0% | 63.3% | 55.9% | 71.8% | 59.6% | 12.2 pts | finance > tech > education > healthcare | yes |
| 0.60 | 13,557 | 76.2% | 66.7% | 63.0% | 55.6% | 71.4% | 59.3% | 12.1 pts | finance > tech > education > healthcare | yes |
| 0.65 | 13,388 | 75.6% | 66.4% | 62.7% | 55.3% | 71.0% | 59.0% | 12.0 pts | finance > tech > education > healthcare | yes |
| 0.70 | 13,171 | 74.9% | 66.0% | 62.1% | 54.8% | 70.5% | 58.4% | 12.0 pts | finance > tech > education > healthcare | yes |
| 0.75 | 12,922 | 74.4% | 65.4% | 61.7% | 53.8% | 69.9% | 57.8% | 12.1 pts | finance > tech > education > healthcare | yes |
| 0.80 | 12,668 | 73.8% | 64.8% | 61.3% | 53.2% | 69.3% | 57.3% | 12.1 pts | finance > tech > education > healthcare | yes |
| 0.85 | 12,353 | 73.1% | 63.8% | 60.3% | 52.2% | 68.5% | 56.3% | 12.2 pts | finance > tech > education > healthcare | yes |
| 0.90 | 11,910 | 71.6% | 62.5% | 59.4% | 51.4% | 67.1% | 55.4% | 11.7 pts | finance > tech > education > healthcare | yes |

Holds at 9 of 9 thresholds.
