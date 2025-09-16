# Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL
Data analytics project predicting player market dynamics in the EPL using FIFA-23, FPL, and Transfermarkt data. Applied clustering, classification, and regression to identify player archetypes, forecast transfer likelihood, and estimate market valuations.
## Overview
EPL transfer market valuations are rising rapidly, while many clubs face declining profitability. Clubs risk overpaying for players whose on-pitch impact does not justify their cost. This project demonstrates how analytics can reduce uncertainty in valuations and support more evidence-based transfer decisions. This project develops a player valuation model by integrating **skill attributes**, **in-game performance**, and **transfer market data**.  
It answers three core questions:
1. What distinct player archetypes can be identified based on their performance, overall rating, and market valuation? (Clustering)
2. Which players are the most likely to be transferred in the next season based on their performance, overall rating, and market valuation? (Classification)
3. Which factor is the best predictor for a player’s market valuation in the future? (Regression)

The analysis combines **FIFA-23 ratings**, **Fantasy Premier League metrics**, and **Transfermarkt data**, applying clustering, classification, and regression techniques. Models include **K-medoids**, **OPTICS**, **Decision Tree**, **Naïve Bayes**, and both **linear and non-linear regression**.
## Data Sources
1. **FIFA-23 Ratings** → player skill attributes [(dataset)](https://www.kaggle.com/datasets/sanjeetsinghnaik/fifa-23-players-dataset)
2. **Fantasy Premier League (FPL)** → in-game performance metrics [(dataset)](https://www.kaggle.com/datasets/meraxes10/fantasy-premier-league-dataset-2022-2023)
3. **Transfermarkt** → transfer activity and market values [(dataset)](https://www.kaggle.com/datasets/davidcariboo/player-scores?select=player_valuations.csv)
## Data Preprocessing
### Descriptive Statistics
![Descriptive Statistics](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/tables/Descriptive%20Statistics.jpg)

The table provides a statistical overview of key metrics for a dataset related to FIFA ratings, playing time, goal/assist contributions, fantasy scores, and market values. While the mean FIFA rating of 73.96 suggests a moderate average performance, the wide range (50–91) and negative skewness (-0.71) indicate a concentration of high-performing players. Minutes played, with a mean of 1209.62 and skewness of 0.44, shows a slight tendency toward players with higher playtime, though the large standard deviation (1095.13) highlights variability. Goal/assist contributions (GAContribution) and fantasy scores reveal highly uneven distributions, evident from their extreme skewness (2.98 and 0.57) and kurtosis values (12.26 and 0.79), indicating a few outliers dominate these metrics. Market value demonstrates significant disparity, with a mean of ~20.5M but a staggering range from 25K to 180M, coupled with a high positive skew (2.07) and kurtosis (6.33), reflecting an uneven distribution skewed toward highly valued players. Overall, the data suggests considerable variability and outlier influence, necessitating careful handling for robust analysis and meaningful insights. The distribution and shape interpretations can be visible in the pairplot.
### Correlation Analysis
![Pairplot](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Pairplot.png)

The pairplot and heatmap below reveal strong correlations: FantasyScore-MinutesPlayed (0.76), FIFARating-MarketValue (0.60), and FantasyScore-GAContribution (0.67). Players with higher minutes generally accumulate better fantasy scores due to more game time contributing to performance metrics. FIFA ratings strongly align with market values, as ratings encapsulate skill, consistency, and market demand. G/A contributions significantly boost fantasy scores, reflecting their direct impact on game outcomes. These trends underscore how playing time, performance metrics, and player marketability interconnect in real-world football analytics.
![Heatmap](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Correlation%20Heatmap.png)
### Outlier Analysis
![Boxplot1](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/FIFA%20Rating%20Boxplot.png) ![Boxplot2](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Minutes%20Played%20Boxplot.png) ![Boxplot3](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/GA%20Contribution%20Boxplot.png) ![Boxplot4](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Fantasy%20Score%20Boxplot.png) ![Boxplot5](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Market%20Value%20Boxplot.png)

Significant outliers are evident in GAContribution, FantasyScore, and MarketValue, representing players like Erling Haaland and Mohammad Salah, who perform exceptionally well. In contrast, FIFARating shows fewer outliers, primarily on the lower end, reflecting players with poor ratings. To manage these outliers effectively, algorithms less sensitive to their influence are strongly recommended.
### Categorical Feature Analysis
![Barplot1](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Age%20Group%20Bar%20Chart.png) ![Barplot2](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Position%20Bar%20Chart.png) ![Countplot](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Position%20vs%20Age%20Group.png)

The bar charts above reveal that the dataset predominantly comprises players aged 22–25, with midfielders being the most represented position. The count plot further highlights that player across all positions, except goalkeepers, are primarily within the 22–25 age range. This suggests a preference for more experienced goalkeepers. Midfielders under 21 appear favoured, likely due to forwards and defenders bearing greater responsibilities. The Sankey plot indicates that midfielders under 21 also accounted for most transfers.
![Sankey Chart](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Relation%20between%20Position%2C%20Age%20Group%20and%20Transfer%20Status.png)
### Proximity Analaysis
![Dissimilarity Matrix](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Dissimilarity%20Matrix%20Heatmap.png)

The dissimilarity matrix for mixed types is applied here to differentiate players from the upper and lower classes. Cosine similarity outperforms other metrics due to its invariance to highly scaled features and robustness in high-dimensional spaces, making it more suitable for clustering. Jaccard is used for categorical features to address their specific characteristics effectively.
## Clustering
### K-medoids Clustering
By assigning participants to particular clusters, clustering algorithms help uncover unnoticed trends, categorise players based on similarities in important attributes, and forecast future behaviour. K-medoids and OPTICS are employed for clustering in this report. K-medoids provide sturdiness against outliers and efficiently handle mixed data types by clustering around actual data points. Additionally, it works well with non-Euclidean proximities that are relevant to this dataset. On the other hand, OPTICS emphasises concentrated groupings of players who are similar by creating clusters based on density. In datasets with numerous numerical features and outliers, it excels in noise reduction, an area that K-medoids do not address.

![Elbow and Silhouette](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Elbow%20Chart%20and%20Silhouette%20Index.jpg)

Cosine proximity is superior at handling numerous features of different magnitudes, as shown by the comparison of Average WCSS and Silhouette Index across different proximity measures. The high Silhouette Index scores and low Average WCSS values imply that cosine proximity produces clusters that are more cohesive and distinct.

![Val K-medoids](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/tables/Validity%20Indices%20for%20different%20K.jpg)

Although K=2 produced relatively better outcomes across all indices, the elbow chart shows K=3 as the most suited cluster count. Additionally, restricting the research to two clusters would make it more difficult to distinguish top-performing players from average players. As a result, choosing K=3 is more in line with the goal of making a more subtle distinction.
### Optics Clustering
![Val Optics](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/tables/Validity%20Indices%20for%20different%20paramters.jpg)

Cosine proximity performs better in this situation, much like K-medoids do, since the model clusters more individuals with a relatively better Silhouette Index. The iterations suggest that attaining a higher Silhouette Index requires a trade-off between the number of grouped members. By changing the minimum points between 30-60, these iterations were carried out across a range of proximity measurements. Higher values would yield a single cluster, while lower values would result in excessive clusters. The iteration with MinPts=45 and Eps=0.01 is considered ideal for this report, because it accounts for 393 members (68% of the sample) and produces an acceptable Silhouette Index of 0.59.

![Reachability Plot](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Reachability%20Plot.png)

Three separate clusters are represented by the pits seen in the reachability figure, with one cluster having a noticeably larger number of data points. This implies that a significant percentage of underperforming players were successfully found and filtered by the model.

### Comparison
![Pairplot Clustering](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Market%20value%20vs%20other%20numericals%20correlation.jpg)

A visual comparison of the two models' clusters demonstrates that OPTICS performs better in this scenario, with more distinct distributions compared to the overlapping clusters in K-medoids. Additionally, OPTICS effectively reduces visible outliers, enhancing the clarity and reliability of the clustering results.
## Classification
Classification techniques are instrumental in identifying patterns among transferred players, enabling the development of a model to estimate the likelihood of future player transfers. Validity tests conducted using the available data evaluate the robustness of each model. This report employs Decision Tree and Naïve Bayes algorithms for classification.

![Val Classification](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/tables/Validity%20Indices%20for%20Classification.jpg)

Cross-validation tests were used to assess four classification models to reduce bias and avoid the risk of overfitting. Based on evaluation indices, Decision Tree and Naïve Bayes performed better than the others. Because of its ability to efficiently handle high-dimensional datasets with a variety of data kinds, Decision Tree was beneficial considering the dataset's inclusion of numerous independent features with different data types. In the same manner, Naïve Bayes did well by effectively managing categorical variables and applying probabilistic reasoning under the presumption of robust feature independence. The relative success of both models in player classification is demonstrated by their larger Area Under Curve (AUC) ratings. While recall quantifies the percentage of real cases that are successfully identified, precision emphasises the percentage of accurate predictions. Even though SVM has a good recall score, its dependability is called into question by its remarkably low AUC value of 0.5, which is close to random guessing. 

![ROC and PC](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/ROC%20Curve%20and%20Performance%20Curve.jpg)

The trade-off between players who are accurately and wrongly identified as transferred—is depicted by the ROC curve. A model should ideally have a curve approaching the upper-left corner, signifying high specificity and sensitivity. In this instance, both curves stay only marginally above the diagonal, indicating that both models' predictive power is moderately low. The trade-off between Precision and Recall is represented by the Performance curve. As more real transferred players are found, Precision falls since there are more false positive predictions, according to the performance curve's downward trend. The limitations of both models in balancing these measures are highlighted by this trade-off, especially in situations where high prediction confidence is required.

![Confusion Matrix](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/tables/Confusion%20Matrices.jpg)

Type-II errors are more common than Type-I errors according to the confusion matrix. Accurately identifying transfers requires minimising Type-II errors since incorrectly classifying them compromises the model's utility. Retained players are more accurately predicted by both models, indicating an asymmetry in predictive power. This suggests that the chosen features might not accurately capture the complexities of transferability. In conclusion, the classification models' poor performance calls for better feature selection or alternative approaches.
## Predictive Modelling
![Matrix Plot](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Matrix%20Plot.jpg)

Regression analysis can formulate equations that can predict the market value and sort the better predictors. For comparing the features for predicting MarketValue, numeric features (FIFARating, MinutesPlayed, GAContribution, FantasyScore) are used. The matrix plot shows that the relationships for MinutesPlayed, FantasyScore, and GAContribution appear somewhat linear beyond a certain threshold, while FIFARating follows an exponential trend. Consequently, this report utilises both linear and non-linear regression approaches to compare results and develop a more accurate model for specific features.
### Linear Regression
![LR Table](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/tables/Linear%20Regressionn%20Results.jpg)

Linear regression was performed using backward elimination to enhance the model by removing features with minimal impact. The initial R-squared value indicates that the features explain 55% of the variance, while the low Durbin-Watson statistic suggests a positive correlation among the residuals. Additionally, the NPP chart's heavy tails indicate the presence of extreme outliers in the residuals. The p-values for MinutesPlayed (0.13) and FantasyScore (0.20) exceed 0.05, suggesting that the null hypothesis (H₀: β₂ = β₄ = 0) cannot be rejected for these features, rendering them insignificant in predicting market value. VIF scores below 5 indicate no multicollinearity, confirming the independence of each feature. The improved model shows no significant differences in the metrics, further supporting FIFARating and GAContribution as more reliable predictors. 

**Linear Regression Equation: MarketValue = -70741889.214 + 1148199.796*FIFARating +1988367.414 * GAContribution**

![NP Plot](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/NPP%20Plot.png)
### Non-linear Regresssion
![Val NLR](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/tables/Non-linear%20Regression%20Results.jpg)

Non-linear regressions were applied to individual features to further determine the best predictors. After running multiple functions on each feature, it is clear that FIFARating yields the highest R-squared value (0.579) using both exponential and growth functions, making it the most reliable predictor. The curve fit below demonstrates a perfect match with the data, while the NPP chart shows a similar pattern to the linear regression. FantasyScore has comparatively higher scores, indicating that linear regression was unsuitable for it, while MinutesPlayed remains insignificant even after applying non-linear regression.

**Exponential Regression Equation: MarketValue = 6043.55 * exp(0.105894 * FIFARating)**

![Curve Fit and NPP](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Curve%20Fit%20and%20NPP.jpg)
## Conclusion
### Player Archetypes
![Archetypes](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/tables/Player%20Archetypes.jpg)

Both the clustering methods distinguish ‘Fringe Players’ (low-performers) well from other archetypes ‘Star Performers’ (top-performers) and ‘Team Anchors’ (average-performers). The IQR of the respective archetypes, highlights a better range for assigning players because of the overlapping range. Improved in-game performance indicators like MinutesPlayed, GAContribution, and FantasyScores set top-performers apart from mediocre players, even when ratings and valuations for these players may overlap. In comparison, OPTICS produces clusters with better silhouette scores, as it keeps players closely related by reducing noise. For instance, Erling Haaland was excluded due to his game-changing statistics, while Cristiano Ronaldo was excluded for his underwhelming performance and limited gametime. This suggests that to better understand a player's behaviour, they can first be assigned to an initial group through K-medoids. Subsequently, OPTICS, combined with feature-wise analysis, would further validate the assignment by identifying whether the player is an outlier.

![Cluster-wise Divisions](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/tables/Cluster-wise%20Age%20Groups%20and%20Positions%20Distribution.jpg)

Analysing the OPTICS clusters with the categorical features, it can refer that most of the Fringe Players belong to ’21 and under’ age group as they don’t get to perform well comparatively due to lack of experience and gametime. Most Star Players are from 22-29 age, highlighting the peak age for players to perform well. Team Anchors includes most of defenders, which might be due to their minimum participation in goals and assists, and recent increases in their market valuations.
### Transfer Prospects
![Cluster-wise Transfers and Market Value](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Cluster-wise%20Transfers%20and%20Market%20Value.jpg)

Combining cluster outputs with actual transfer data reveals that fringe players are transferred more frequently, often due to clubs loaning out players with limited game time to maximise their potential value. However, when considering the market values of transferred players, it becomes evident that clubs prioritise investing in star performers. Despite fewer transfers, the market valuations of players in the star performer cluster significantly surpass those of other clusters, reflecting their higher investment appeal.

![Cluster-wise Predictions](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Cluster-wise%20Predictions.jpg)

Using classification algorithms, transfer predictions were not highly accurate, as shown in the chart. However, these algorithms performed better in predicting retentions. Notably, the Naïve Bayes model proved relatively weak, as it failed to predict transfers outside the fringe player category. On a positive note, the models effectively reduced type-II errors, minimising false predictions of retentions. This improvement may stem from players with false predictions being identified as noise by the OPTICS model. These findings suggest that combining clustering and classification algorithms could help clubs avoid overlooking players likely to be transferred, enhancing overall prediction accuracy.

### Market Value Estimations
![Clusterwise Market Value vs FIFA Rating](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Cluster-wise%20Market%20Value%20vs%20FIFA%20Rating.jpg)

Regression analysis reveals FIFA ratings as the most reliable predictor of market value. These ratings are calculated based on a player's real-world performance, skills, and attributes such as pace, dribbling, shooting, and defending, which are weighted and aggregated into an overall score. By providing a standardised benchmark for comparing players' abilities, FIFA ratings help assess market value by identifying potential, consistency, and role suitability. Managers can assign a targeted player to the appropriate cluster and use the interquartile range (IQR) to estimate their potential transfer value.

## Limitations and Recommendations
This research is limited EPL players and does not differentiate between domestic and international transfers. As such, it lacks generalizability to other leagues where priorities may differ. A narrower scope, focusing on a single position across leagues as done in some literature, could provide a more comparative perspective. Traits specific to defenders and goalkeepers are not considered in this report, despite their importance in assessments. Additionally, relying on goal and assist contributions skews the data, misinterpreting some players as weaker performers. Assessing players based on position-specific traits with proper weightings, as shown in other literature, would improve accuracy. Team performance and tactics strongly influence player contributions, as managers employ different strategies to win. Some papers addressed this by separating variables into internal (player-specific) and external (team and managerial) factors for distinct analysis. The research does not factor in club finances, such as the ability to support increased wages or extend contracts. These are critical to player valuations, as some clubs cannot afford high-value players. Literature addressing this used the Nash bargaining model to account for financial constraints.
The projects's strengths are multidisciplinary approach, combining data from diverse sources and applying varied analytical techniques to addresses the key gaps in existing literature. This research offers a foundation for advancing player valuation analytics and paves the way for improved transfer strategies and broader football market studies.
