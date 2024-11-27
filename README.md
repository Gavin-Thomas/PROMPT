

          
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


## IADL Women Score Distribution Relative to Men

### Graphs
![Combined Normalized IADL Scores MEN](https://github.com/Gavin-Thomas/PROMPT/blob/main/images/IADL%20MEN.png?raw=true)
   This graph shows the density of normalized functional scores for men, grouped by definite cognitive categories.

![Combined Normalized IADL Scores WOMEN](https://github.com/Gavin-Thomas/PROMPT/blob/main/images/IADL%20WOMEN.png?raw=true) 
   This graph highlights the distribution of normalized functional scores for women across definite cognitive categories.

#### Men
- **Definite Normal**:
  - Scores peak near **100%**, indicating high functional independence.
- **Definite MCI**:
  - Scores are broadly distributed between **50–80%**, reflecting moderate independence.
- **Definite Dementia**:
  - Scores cluster in the lower range (**0–40%**), signifying severe impairment.

#### Women
- **Definite Normal**:
  - Similarly concentrated near **100%**, reflecting high functional ability.
- **Definite MCI**:
  - Distributed across the mid-range, showing consistent patterns with men.
- **Definite Dementia**:
  - Scores align with men, clustering in the **0–40%** range.
 
WITH THE SCORING, I NOTICED THAT BOTH MEN AND WOMEN WERE SCORED THE SAME WAY. SO THERE IS NO DISCREPANCY IN SCORING.

# MoCA and MMSE Total Scores by Cognitive Categories

## Overview
This section examines the distributions of **MoCA Total Scores** and **MMSE Total Scores** across cognitive categories (Definite Normal, Definite MCI, and Definite Dementia). These assessments provide critical insights into cognitive performance, with higher scores reflecting better cognition.

### Key Findings
#### **MoCA Total Scores**:
- **Definite Normal**:
  - Scores peak in the range of **27–30**, indicating intact cognitive function.
- **Definite MCI**:
  - Broadly distributed across the mid-range (**20–25**), consistent with mild impairments.
- **Definite Dementia**:
  - Concentrated in the lower range (**0–15**), reflecting significant cognitive decline.

#### **MMSE Total Scores**:
- **Definite Normal**:
  - Peaks near **28–30**, aligning with preserved cognition.
- **Definite MCI**:
  - Scores mostly range between **24–27**, showing mild impairments.
- **Definite Dementia**:
  - Scores are heavily clustered in the **0–20** range, indicating severe cognitive impairment.

---

## Figures

### Figure 1: MoCA Total Scores by Cognitive Categories
**Description**: The distribution of MoCA scores highlights distinct patterns across cognitive categories. Definite Normal individuals cluster at the upper range, while Definite Dementia scores are concentrated at the lower end.
![MoCA Total Scores](path/to/moca_scores.png)

### Figure 2: MMSE Total Scores by Cognitive Categories
**Description**: The MMSE scores show similar patterns to the MoCA, with clear differentiation between cognitive categories. Definite Normal scores peak near 30, while Definite Dementia clusters at lower scores.
![MMSE Total Scores](path/to/mmse_scores.png)

---

## Comparison: Men vs Women

### MoCA Total Scores
#### **Men**:
- Definite Normal men cluster in the range of **28–30**.
- Definite Dementia scores for men are concentrated in the **0–15** range.
- Men show slightly less variability compared to women in Definite MCI.

#### **Women**:
- Definite Normal women also peak near **28–30**.
- Women in the Definite Dementia category show more spread in the lower range (0–20).

### MMSE Total Scores
#### **Men**:
- Definite Normal men cluster in the range of **28–30**, similar to women.
- Definite Dementia scores are sharply concentrated near **0–15** for men.
  
#### **Women**:
- Definite Normal scores peak similarly near **28–30**.
- Women in Definite Dementia categories show slightly greater spread than men, indicating more variability in scores.

### Figures

**Figure 3. MoCA Total Scores by Gender**
MoCA scores for men and women show similar trends, with distinct peaks for Definite Normal near 30 and broad distributions for Definite MCI. Minor gender differences are observed in variability for Definite Dementia.

**Figure 4: MMSE Total Scores by Gender**
**Description**: MMSE scores align closely for men and women, with both groups showing clear separation between cognitive categories. Minor differences in variability are observed for Definite MCI and Definite Dementia.

---

## Implications
These results demonstrate the utility of MoCA and MMSE scores in distinguishing cognitive categories. Gender differences, while minor, may provide insights into variability within specific categories, particularly in the Definite Dementia group. These findings can guide tailored interventions and assessments.

