# Week of 
---

          
# Final Combined Normalized IADL Scores

## Overview
The **Combined Normalized IADL Scores** provide a unified measure of functional ability, integrating both **Old IADL** and **New IADL** scores into a single metric. This ensures comparability across individuals assessed using different scales while maintaining consistency in interpreting higher scores as better functional independence.

### Key Steps I Took:
1. **Normalization of New IADL Scores**:
   - Raw scores (0–23) were normalized using: raw score / 23 *100
   - This scales the New IADL scores to a percentage (0–100%), where 100% reflects perfect function.

2. **Normalization of Old IADL Scores**:
   - Raw scores (7–21) were normalized using: raw score / (21-7) * 100
   - Here, 7 (perfect score) maps to **100%**, and 21 (worst score) maps to **0%**, reversing the original scale.

3. **Combining Scores**:
   - New IADL scores were prioritized when available.
   - If New IADL scores were missing, Old IADL scores were used.

4. **Visualization**:
   - A histogram with kernel density estimation (KDE) was created to display the distribution of the combined normalized scores across cognitive categories.
![Combined Normalized IADL Scores](https://github.com/Gavin-Thomas/PROMPT/blob/main/images/IADL.png?raw=true)
**Figure 1. Distribution of Combined Normalized IADL Scores**  
This histogram shows the distribution of combined normalized IADL scores across cognitive categories (Definite Normal, MCI, Dementia). Scores are normalized to a 0–100% scale, where higher values indicate better function. New IADL scores were prioritized when available, with Old IADL scores used otherwise. Definite Normal individuals cluster near 100%, MCI individuals are spread across 50–80%, and Dementia individuals are concentrated in the lower range (0–40%).

## Interpretation of the Histogram
The histogram of combined normalized IADL scores shows:
- **Definite Normal** individuals cluster near **100%**, reflecting high functional independence.
- **Definite MCI** individuals are broadly distributed, with most scores between **50–80%**, consistent with moderate functional ability.
- **Definite Dementia** individuals cluster at the lower end (0–40%), indicating severe impairment.

## Summary Table

| Metric                  | Old IADL         | New IADL         | Combined Normalized IADL |
|-------------------------|------------------|------------------|--------------------------|
| **Count**              | 681              | 649              | 1329                     |
| **Mean**               | 9.85             | 17.15            | 77.16                    |
| **Standard Deviation** | 3.31             | 6.07             | 25.13                    |
| **Minimum**            | 7                | 0                | 0.00                     |
| **25th Percentile**    | 7                | 14               | 64.29                    |
| **Median**             | 9                | 19               | 85.71                    |
| **75th Percentile**    | 12               | 23               | 100.00                   |
| **Maximum**            | 21               | 23               | 100.00                   |

