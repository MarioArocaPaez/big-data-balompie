import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the clustered data
data = pd.read_csv('betis_clusters.csv')

# Select features and the target variable
features = ['Performance_Gls', 'Performance_Ast', 
            'Progression_PrgC', 'Progression_PrgP', 'Progression_PrgR', 
            'Expected_xG', 'Expected_xAG']
target = 'Cluster'

# Split the data into training and testing sets
X = data[features].fillna(0)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the model for future use
joblib.dump(model, 'player_cluster_predictor.pkl')
print("Model saved as 'player_cluster_predictor.pkl'")

# Function to predict the cluster for a new player
def predict_new_player(stats):
    """
    Predicts the cluster for a new player based on their stats.

    :param stats: A dictionary containing the player's stats.
                  Example: {'Performance_Gls': 5, 'Performance_Ast': 3, 'Progression_PrgC': 10, ...}
    :return: Predicted cluster (int).
    """
    stats_df = pd.DataFrame([stats])
    stats_df = stats_df[features].fillna(0)  # Ensure correct order and fill missing values
    prediction = model.predict(stats_df)
    return prediction[0]

# Example: Predict for a new player
new_player_stats = {
    'Performance_Gls': 3,
    'Performance_Ast': 2,
    'Progression_PrgC': 20,
    'Progression_PrgP': 50,
    'Progression_PrgR': 30,
    'Expected_xG': 2.5,
    'Expected_xAG': 1.8
}
predicted_cluster = predict_new_player(new_player_stats)
print("Predicted Cluster for New Player:", predicted_cluster)
