import soccerdata as sd
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

# Enable debug-level logging to track program flow
logging.basicConfig(level=logging.DEBUG)

# Initialize the FBref scraper to fetch historical data for La Liga
fbref = sd.FBref(leagues=['ESP-La Liga'])

# Download player statistics for all available seasons
player_stats = fbref.read_player_season_stats(stat_type="standard").reset_index()

# Filter the data to include only players from Real Betis and create a new DataFrame copy
betis_stats = player_stats[player_stats['team'] == 'Betis'].copy()

# Rename the columns to simplify access (flatten the MultiIndex columns into single-level names)
betis_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in betis_stats.columns]

# Select relevant features (performance metrics) and fill missing values with 0
features = betis_stats[['Performance_Gls', 'Performance_Ast',
                        'Progression_PrgC', 'Progression_PrgP', 'Progression_PrgR',
                        'Expected_xG', 'Expected_xAG']].fillna(0)

# Normalize the feature values to a standard scale (mean=0, variance=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Initialize the K-Means clustering model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Apply the K-Means algorithm to the normalized features and assign each player to a cluster
betis_stats['Cluster'] = kmeans.fit_predict(features_scaled)

# Find the column that contains player names dynamically
player_column = [col for col in betis_stats.columns if 'player' in col.lower()][0]

# Save the results to a CSV file, including features and cluster assignments
betis_stats[['Performance_Gls', 'Performance_Ast',
             'Progression_PrgC', 'Progression_PrgP', 'Progression_PrgR',
             'Expected_xG', 'Expected_xAG', player_column, 'Cluster']].to_csv('betis_clusters_prediction.csv', index=False)
print("Results saved to 'betis_clusters_prediction.csv'")
