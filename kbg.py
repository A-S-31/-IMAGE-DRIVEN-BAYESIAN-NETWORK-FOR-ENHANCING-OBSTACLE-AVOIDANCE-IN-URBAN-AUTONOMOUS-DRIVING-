import pandas as pd
from pgmpy.estimators import K2Score, HillClimbSearch
from pgmpy.models import BayesianModel
import networkx as nx
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('C:/Users/aarsh/OneDrive/Desktop/PGM/asia_cancer.csv')  # Replace with your dataset

# Define the scoring function (K2 score)
scoring = K2Score(data)

# Initialize the Hill Climb Search algorithm
hc = HillClimbSearch(data)

# Find the best structure
best_model = hc.estimate(scoring)

# Print the learned structure
print("Learned Bayesian Network Structure:")
print(best_model.edges())

model = BayesianModel(best_model)
G = nx.DiGraph(model.edges())

pos = nx.spring_layout(G) 
nx.draw(G,pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', edge_color='black', arrows=True, arrowstyle='-|>', arrowsize=20)
plt.show()

