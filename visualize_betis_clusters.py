import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file with the player clusters
betis_clusters = pd.read_csv('betis_clusters.csv')

# Load the original features and merge them with the cluster data
betis_stats = pd.read_csv('betis_clusters.csv')  # Add the full stats file if needed
all_stats = betis_stats.copy()  # Ensure your full stats are aligned with clusters

# Merge cluster data with all stats
betis_with_clusters = pd.merge(betis_clusters, all_stats, on='player_', how='left')

# Use the correct Cluster column after the merge
betis_with_clusters['Cluster'] = betis_with_clusters['Cluster_x']

# Debug the merged DataFrame
print(betis_with_clusters.head())
print(betis_with_clusters.columns)

# Ensure only numeric columns are included in the calculation
numeric_columns = betis_with_clusters.select_dtypes(include=['float64', 'int64']).columns

# Calculate the average statistics for each cluster
cluster_summary = betis_with_clusters.groupby('Cluster')[numeric_columns].mean()

# Display the cluster summary as a simple table
print("\nCluster Summary:")
print(cluster_summary)

# Visualize the cluster summary with a bar chart
plt.figure(figsize=(12, 8))
cluster_summary.plot(kind='bar', figsize=(12, 8), title="Cluster Summary for Betis Players")
plt.ylabel("Average Metric Values")
plt.xlabel("Cluster")
plt.xticks(rotation=0)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
plt.tight_layout()

# Save the chart as an image
plt.savefig('betis_cluster_summary.png', format='png', dpi=300)
print("Cluster summary chart saved as 'betis_cluster_summary.png'.")

# Show the plot
plt.show()
