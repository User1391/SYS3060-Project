# yfinance use from https://medium.com/@kasperjuunge/yfinance-10-ways-to-get-stock-data-with-python-6677f49e8282
import yfinance as yf
import statistics as stat
import pydtmc
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# get our initial data
data = yf.download("AAPL", start="2018-01-01", end = "2024-01-01")
data = data.drop(columns=["High", "Low", "Adj Close", "Volume"])

# get the daily percentage change from open to close
pchange = []
for idx, row in data.iterrows():
    pchange.append(((row["Close"] - row["Open"])/row["Open"]))

# print relevant information for constructing the Markov chain 
print("Mean:", stat.mean(pchange))
print("Median:", stat.median(pchange))
print("Variance:", stat.variance(pchange))
print("Standard Deviation:", stat.stdev(pchange))

# based off of this information, we should do 1pp increments, from -3 to 3
# analysis and node selection inspired from https://kth.diva-portal.org/smash/get/diva2:1823899/FULLTEXT01.pdf
labeled_change = []
for val in pchange:
    if (val < -0.03):
        labeled_change.append("D3")
    elif (val < -0.02):
        labeled_change.append("D2")
    elif (val < -0.01):
        labeled_change.append("D1")
    elif (val < 0):
        labeled_change.append("D0")
    elif (val < 0.01):
        labeled_change.append("U0")
    elif (val < 0.02):
        labeled_change.append("U1")
    elif (val < 0.03):
        labeled_change.append("U2")
    else:
        labeled_change.append("U3")

# list of possible states
states = ["D3", "D2", "D1", "D0", "U0", "U1", "U2", "U3"]
print("")
print("Markov Chain First-Order Assessment")
print(pydtmc.assess_first_order(states, labeled_change))

# count occurrences
print("\nOccurrences:")
for state in states:
    print(state + ": " + str(labeled_change.count(state)))

# create a markov chain
mkc = pydtmc.MarkovChain.fit_sequence("mle", states, labeled_change)

# get the transition matrix
transition_matrix = pd.DataFrame(mkc.to_matrix())

# add the state names
transition_matrix.columns = states
transition_matrix.index = states

print("")
print("Transition Matrix:")
print(transition_matrix)

print("")
ssdf = pd.DataFrame(mkc.pi)
ssdf.columns = states
ssdf.index = ["Steady State"]
print(ssdf)

# create the graph
gr = mkc.to_graph()

# clean up the edge values
edge_vals = {}
for out_edge in states:
    for in_edge in states:
        edge_vals[(out_edge, in_edge)] = round(gr.edges[(out_edge, in_edge)]["weight"], 4)

pos = nx.spring_layout(gr)
nx.draw_networkx_nodes(gr, pos, node_size = 500)
nx.draw_networkx_labels(gr, pos)
nx.draw_networkx_edges(gr, pos, edgelist=gr.edges())
nx.draw_networkx_edge_labels(gr, pos, edge_vals)
plt.show()

print("")

