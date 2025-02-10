# Link Prediction in Country Graph

This project focuses on link prediction in a graph where nodes represent countries and edges represent connections based on the last letter of one country's name matching the first letter of another country's name. We compare the performance of two models: Node2Vec and a Graph Neural Network (GNN).

## Table of Contents
- [Introduction](#introduction)
- [Node Features](#node-features)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Logic behind Unsupervised Learning](#unsupervised-training-objective)
- [Conclusion](#conclusion)
- [Usage](#usage)

## Introduction
Link prediction is a fundamental task in network analysis, aiming to predict the existence of edges between nodes in a graph. In this project, we use a dataset of country names to create a graph and predict potential connections using Node2Vec and GNN models.
## Node Features
I have just decided to go for the simple approach of using the first and the last letters for this Task as those are not complex and the corpus is small that they can be used without any large computational overhead,However if we were dealing with a larger corpus then could have used some NLP techniques as Node features for it instead.
## Data Preprocessing
The data preprocessing involves:
1. Loading country names from a file.
2. Creating a graph where nodes are countries and edges exist if the last letter of one country's name matches the first letter of another country's name.
3. Generating node features based on the first and last letters of the country names and the length of the country name.

## Model Training
### Node2Vec
Node2Vec is an algorithm that generates node embeddings by simulating random walks on the graph. The steps include:
1. Creating a directed graph from the country data.
2. Training the Node2Vec model to generate embeddings for each node.
3. Evaluating the model using ROC AUC and Average Precision (AP) scores.

### GNN
The GNN model uses graph convolutional layers to learn node embeddings. The steps include:
1. Creating node features and edge indices.
2. Splitting the data into training, validation, and test sets.
3. Training the GNN model using the training set.
4. Evaluating the model on the validation and test sets.

## Evaluation
The models are evaluated based on:
- ROC AUC: Measures the ability of the model to distinguish between positive and negative edges.
- Average Precision (AP): Measures the precision-recall trade-off.

## Results
The results of the models are as follows:

### Model Performance
| Metric               | Node2Vec | GNN    |
|----------------------|----------|--------|
| Test AUC             | 0.7284   | 0.8076 |
| Training Time (s)    | 85.53    | 0.28   |

### Model Characteristics
**Node2Vec:**
- Uses pure graph structure
- Fast training
- Memory efficient for sparse graphs

**GNN:**
- Uses both graph structure and node features
- Can capture complex patterns
- More suitable for inductive learning

## Unsupervised Training Objective
The reason for Unsupervised Training is that it is helps to identify patterns previously unseen by the manual annotator and helps identify sub-structures in the graph similar to each other which can help with observation of patterns.

## Conclusion
Based on the results, the GNN model performs better for this specific task, achieving a higher Test AUC score compared to Node2Vec. The GNN model's ability to utilize both graph structure and node features allows it to capture more complex patterns in the data.
## Usage
To run the link prediction task, follow these steps:
1. Ensure you have the required dependencies installed.
2. Place the country names in a file named `countries.txt`.
3. Run the `linkprediction.py` script:
    ```bash
    python linkprediction.py
    ```

This will load the data, train both models, and print the evaluation results.

