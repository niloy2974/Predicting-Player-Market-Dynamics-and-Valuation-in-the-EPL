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
1. **FIFA-23 Ratings** → player skill attributes [dataset](https://www.kaggle.com/datasets/sanjeetsinghnaik/fifa-23-players-dataset)
2. **Fantasy Premier League (FPL)** → in-game performance metrics [dataset](https://www.kaggle.com/datasets/meraxes10/fantasy-premier-league-dataset-2022-2023)
3. **Transfermarkt** → transfer activity and market values [dataset](https://www.kaggle.com/datasets/davidcariboo/player-scores?select=player_valuations.csv)
