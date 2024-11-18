# **Big Data Balompié: Machine Learning with Real Betis**

Welcome to **Big Data Balompié**, a project where we use Machine Learning (ML) to analyze Real Betis players' performance and clusters based on their play style! Whether you're new to Machine Learning or a football fan curious about how data can reveal player roles, this project is for you.

---

## **Project Overview**

This project processes historical data of Real Betis players, applies ML techniques to uncover player roles, and provides tools for visualization and prediction. The project is split into three scripts for clarity and modularity:

1. **`process_betis_data.py`**: Scrapes player data, applies clustering, and saves the results.
2. **`visualize_betis_clusters.py`**: Visualizes the clusters and provides insights.
3. **`predict_player_cluster.py`**: Builds a predictive model to classify players into clusters.

---

## **How It Works**

### **1. Data Processing**
The first script collects data on Real Betis players, including:
- Goals scored (`Performance_Gls`),
- Assists (`Performance_Ast`),
- Progression metrics (`Progression_PrgC`, `Progression_PrgP`, `Progression_PrgR`),
- Expected goals (`Expected_xG`) and assists (`Expected_xAG`).

It applies **K-Means clustering** to group players into clusters based on similar play styles. The results are saved as `betis_clusters.csv`.

---

### **2. Visualization**
The second script analyzes the clusters and generates:
- A **summary table** with average statistics for each cluster.
- A **bar chart** showing the differences between clusters.

For example:
- **Cluster 0**: High goal scorers (likely forwards).
- **Cluster 1**: Balanced players (likely midfielders).
- **Cluster 2**: High progression metrics (likely playmakers or defenders).

---

### **3. Prediction**
The third script uses the clustered data to build a **Random Forest Classifier**, which:
- **Trains** on historical data to predict a player's cluster.
- **Predicts** the cluster for new players based on their stats.

Example:
- Input: `{Goals: 5, Assists: 3, Progression_Passes: 20}`
- Output: `Cluster 1`

---

## **Setup**

### **1. Prerequisites**
Ensure you have Python installed and the following libraries:
- `soccerdata`
- `pandas`
- `scikit-learn`
- `matplotlib`

Install them using:
```bash
pip install soccerdata pandas scikit-learn matplotlib
```
### **2. Clone the Repository**

```bash
git clone https://github.com/MarioArocaPaez/big-data-balompie.git
cd big-data-balompie
```

### **3. Run the Scripts**

#### **Step 1: Data Processing**

Run the first script to scrape data, apply clustering, and save results:

```bash
python process_betis_data.py
````
Check the generated `betis_clusters.csv` file for results.

---

#### **Step 2: Visualization**

Visualize the clusters with:

```bash
python visualize_betis_clusters.py
````
#### **Step 3: Prediction**

Build a predictive model and test it:

```bash
python predict_player_cluster.py
````
Try predicting the cluster for a new player by editing the new_player_stats dictionary.
