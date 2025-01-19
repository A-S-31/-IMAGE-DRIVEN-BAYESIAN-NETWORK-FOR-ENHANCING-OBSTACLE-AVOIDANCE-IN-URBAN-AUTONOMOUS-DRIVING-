import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
days = 3  # Number of time slices
num_patients = 100  # Number of patients

# Possible states
health_states = ['Healthy', 'Sick']
medication_states = ['Adherent', 'Non-Adherent']

# Initialize arrays to store data
health = np.zeros((num_patients, days), dtype=int)
medication = np.zeros((num_patients, days), dtype=int)

# Initial health status (randomly assign healthy or sick)
health[:, 0] = np.random.choice([0, 1], num_patients)  # 0: Healthy, 1: Sick
medication[:, 0] = np.random.choice([0, 1], num_patients)  # 0: Adherent, 1: Non-Adherent

# Transition probabilities
transition_probabilities = {
    (0, 0): [0.9, 0.1],  # Healthy -> Healthy, Sick
    (0, 1): [0.7, 0.3],  # Healthy + Non-Adherent -> Healthy, Sick
    (1, 0): [0.4, 0.6],  # Sick -> Healthy, Sick
    (1, 1): [0.2, 0.8],  # Sick + Non-Adherent -> Healthy, Sick
}

# Simulate health status over time
for t in range(1, days):
    for i in range(num_patients):
        prev_health = health[i, t - 1]
        medication_status = medication[i, t - 1]
        # Determine transition probabilities
        probs = transition_probabilities[(prev_health, medication_status)]
        # Sample new health status based on probabilities
        health[i, t] = np.random.choice([0, 1], p=probs)
        # Randomly assign medication adherence
        medication[i, t] = np.random.choice([0, 1])

# Create DataFrames for easy visualization
df_health = pd.DataFrame(health, columns=[f'Day {i}' for i in range(days)])
df_medication = pd.DataFrame(medication, columns=[f'Day {i}' for i in range(days)])

print("Health Status Over Time:")
print(df_health.replace({0: 'Healthy', 1: 'Sick'}))

print("\nMedication Adherence Over Time:")
print(df_medication.replace({0: 'Adherent', 1: 'Non-Adherent'}))

# Display Conditional Probability Tables (CPTs)
print("\nConditional Probability Tables (CPTs):")
cpt_data = []

for (prev_health, medication), probs in transition_probabilities.items():
    cpt_data.append({
        'Previous Health': 'Healthy' if prev_health == 0 else 'Sick',
        'Medication': 'Adherent' if medication == 0 else 'Non-Adherent',
        'P(Healthy)': probs[0],
        'P(Sick)': probs[1]
    })

cpt = pd.DataFrame(cpt_data)
print(cpt)

# Function to visualize the DBN structure
def visualize_dbn(days):
    # Create a directed graph
    G = nx.DiGraph()

    # Define nodes for health status and medication adherence for each time slice
    for t in range(days):
        G.add_node(f'H_{t}', label='Health Status')
        G.add_node(f'M_{t}', label='Medication Adherence')

        if t > 0:
            # Add edges from previous time slice health to current health
            G.add_edge(f'H_{t-1}', f'H_{t}')
            # Add edges from current medication adherence to current health
            G.add_edge(f'M_{t}', f'H_{t}')

    # Set positions for the nodes
    pos = {}
    for t in range(days):
        pos[f'H_{t}'] = (t, 1)  # Health nodes
        pos[f'M_{t}'] = (t, 0)  # Medication nodes

    # Draw the graph
    plt.figure(figsize=(12, 6))
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold')
    
    # Add labels
    labels = {node: G.nodes[node]['label'] for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title('Dynamic Bayesian Network Structure Over Time')
    plt.axis('off')
    plt.show()

# Call the visualization function
visualize_dbn(days)
