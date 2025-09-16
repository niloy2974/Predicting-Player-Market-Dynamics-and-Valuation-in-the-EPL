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
### Numeric Features
![Descriptive Statistics](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/tables/Descriptive%20stats.jpg)

The table provides a statistical overview of key metrics for a dataset related to FIFA ratings, playing time, goal/assist contributions, fantasy scores, and market values. While the mean FIFA rating of 73.96 suggests a moderate average performance, the wide range (50–91) and negative skewness (-0.71) indicate a concentration of high-performing players. Minutes played, with a mean of 1209.62 and skewness of 0.44, shows a slight tendency toward players with higher playtime, though the large standard deviation (1095.13) highlights variability. Goal/assist contributions (GAContribution) and fantasy scores reveal highly uneven distributions, evident from their extreme skewness (2.98 and 0.57) and kurtosis values (12.26 and 0.79), indicating a few outliers dominate these metrics. Market value demonstrates significant disparity, with a mean of ~20.5M but a staggering range from 25K to 180M, coupled with a high positive skew (2.07) and kurtosis (6.33), reflecting an uneven distribution skewed toward highly valued players. Overall, the data suggests considerable variability and outlier influence, necessitating careful handling for robust analysis and meaningful insights. The distribution and shape interpretations can be visible in the pairplot.

![Pairplot](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Pairplot.png)![Correlation Heatmap](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Correlation%20Heatmap.png)

The pairplot and heatmap reveal strong correlations: FantasyScore-MinutesPlayed (0.76), FIFARating-MarketValue (0.60), and FantasyScore-GAContribution (0.67). Players with higher minutes generally accumulate better fantasy scores due to more game time contributing to performance metrics. FIFA ratings strongly align with market values, as ratings encapsulate skill, consistency, and market demand. G/A contributions significantly boost fantasy scores, reflecting their direct impact on game outcomes. These trends underscore how playing time, performance metrics, and player marketability interconnect in real-world football analytics.

![Boxplot1](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/FIFA%20Rating%20Boxplot.png)

![Boxplot2](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Minutes%20Played%20Boxplot.png)

![Boxplot3](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/GA%20Contribution%20Boxplot.png)

![Boxplot4](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Fantasy%20Score%20Boxplot.png)

![Boxplot5](https://github.com/niloy2974/Predicting-Player-Market-Dynamics-and-Valuation-in-the-EPL/blob/main/visualisations/Market%20Value%20Boxplot.png)

Significant outliers are evident in GAContribution, FantasyScore, and MarketValue, representing players like Erling Haaland and Mohammad Salah, who perform exceptionally well. In contrast, FIFARating shows fewer outliers, primarily on the lower end, reflecting players with poor ratings. To manage these outliers effectively, algorithms less sensitive to their influence are strongly recommended.
