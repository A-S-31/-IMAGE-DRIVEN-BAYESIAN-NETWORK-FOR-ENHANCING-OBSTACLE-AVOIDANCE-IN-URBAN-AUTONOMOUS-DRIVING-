import pandas as pd
from pgmpy.estimators import BicScore
from pgmpy.models import BayesianModel
import networkx as nx
import matplotlib.pyplot as plt

def calculate_bic(data, model):
    """ Calculate the BIC score for the given model. """
    bic = BicScore(data)
    return bic.score(model)

def add_edge(graph, parent, child):
    """ Add an edge to the graph. """
    if parent not in graph:
        graph[parent] = []
    graph[parent].append(child)

def kbges_algorithm(data, max_parents):
    """
    KBGES Algorithm for learning a Bayesian Network structure.
    
    Parameters:
    data: DataFrame - the complete sample data.
    max_parents: int - maximum number of parent nodes allowed.

    Returns:
    graph: dict - the optimal Bayesian network structure as an adjacency list.
    """
    # Step 1: Initialize a boundless graph with nodes
    nodes = data.columns.tolist()
    graph = {}

    # Step 2: Iterate through each node
    for j in range(len(nodes)):
        Xj = nodes[j]
        
        # Step 3: Calculate the initial BIC score
        Vold = calculate_bic(data, BayesianModel(graph))

        # Step 4: While loop for adding edges
        while True:
            best_parent = None
            best_bic = float('-inf')
            
            # Step 5: Evaluate adding each node as a parent
            for i in range(j):
                Xi = nodes[i]
                
                # Check if we haven't exceeded max parents
                if Xi not in graph or len(graph[Xi]) < max_parents:
                    # Add edge temporarily
                    add_edge(graph, Xi, Xj)
                    Vnew = calculate_bic(data, BayesianModel(graph))
                    
                    # Step 6: Check if the new BIC score is better
                    if Vnew > best_bic:
                        best_bic = Vnew
                        best_parent = Xi
                    
                    # Remove the edge for next iteration
                    if Xi in graph and Xj in graph[Xi]:
                        graph[Xi].remove(Xj)

            # Step 7: If no improvement, break the loop
            if best_bic > Vold and (best_parent is not None):
                # Update the graph
                add_edge(graph, best_parent, Xj)
                Vold = best_bic
            else:
                break

    return graph

def visualize_bayesian_network(graph):
    """ Visualize the Bayesian network using networkx and matplotlib. """
    model = BayesianModel(graph)
    G = nx.DiGraph(model.edges())

    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.title("Learned Bayesian Network")
    plt.show()

# Example usage
data_path = 'C:/Users/aarsh/OneDrive/Desktop/PGM/asia_cancer.csv'  # Path to your dataset
data = pd.read_csv(data_path)  # Load the dataset
max_parents = 3  # Set maximum number of parent nodes
optimal_structure = kbges_algorithm(data, max_parents)


pos=nx.spring_layout(optimal_structure)
nx.draw(optimal_structure,pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', edge_color='black', arrows=True, arrowstyle='-|>', arrowsize=20)
plt.show()
