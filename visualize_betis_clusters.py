import pandas as pd
import matplotlib.pyplot as plt
import random

# Load the CSV file with the player clusters
betis_clusters = pd.read_csv('betis_clusters.csv')

# Load the original features and merge them with the cluster data
betis_stats = pd.read_csv('betis_clusters.csv')  # Add the full stats file if needed
all_stats = betis_stats.copy()  # Ensure your full stats are aligned with clusters

# Merge cluster data with all stats
betis_with_clusters = pd.merge(betis_clusters, all_stats, on='player_', how='left')

# Use the correct Cluster column after the merge
betis_with_clusters['Cluster'] = betis_with_clusters['Cluster_x']

# Ensure only numeric columns are included in the calculation
numeric_columns = betis_with_clusters.select_dtypes(include=['float64', 'int64']).columns

# Calculate the average statistics for each cluster
cluster_summary = betis_with_clusters.groupby('Cluster')[numeric_columns].mean()

# Select 5 random players from each cluster
random_players_per_cluster = {}
for cluster in betis_with_clusters['Cluster'].unique():
    players_in_cluster = betis_with_clusters[betis_with_clusters['Cluster'] == cluster]['player_'].drop_duplicates()
    random_players_per_cluster[cluster] = random.sample(list(players_in_cluster), min(10, len(players_in_cluster)))

# Prepare text for displaying random player names in the chart
player_texts = {}
for cluster, players in random_players_per_cluster.items():
    player_texts[cluster] = "\n".join(players)

# Visualize the cluster summary with a bar chart
plt.figure(figsize=(14, 8))
cluster_summary.plot(kind='bar', figsize=(14, 8), title="Cluster Summary for Betis Players")
plt.ylabel("Average Metric Values")
plt.xlabel("Cluster")
plt.xticks(rotation=0)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))

# Add the random player names as text annotations to the graph
for idx, cluster in enumerate(cluster_summary.index):
    plt.text(
        idx,  # Position on x-axis
        cluster_summary.iloc[idx].max() + 5,  # Position slightly above the tallest bar
        f"Players Like:\n{player_texts.get(cluster, 'N/A')}",  # Random players
        ha='center', va='bottom', fontsize=10, color='blue'
    )

plt.tight_layout()

# Save the chart as an image
plt.savefig('betis_cluster_summary_with_players.png', format='png', dpi=300)
print("Cluster summary chart with random players saved as 'betis_cluster_summary_with_players.png'.")

# Show the plot
plt.show()
