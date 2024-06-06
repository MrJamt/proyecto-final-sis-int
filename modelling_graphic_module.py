import matplotlib.pyplot as plt
import networkx as nx
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def get_model_architecture(model):
    architecture = []
    input_dim = None
    for layer in model.layers:
        if isinstance(layer, Dense):
            if input_dim is None:
                input_dim = layer.input_shape[1]
                architecture.append(input_dim)
            architecture.append(layer.units)
    return architecture

def draw_neural_net(model):
    architecture = get_model_architecture(model)
    G = nx.DiGraph()
    layer_sizes = architecture
    num_layers = len(layer_sizes)

    pos = {}
    node_labels = {}
    colors = []

    v_spacing = 1  # Vertical spacing between layers
    h_spacing = 1  # Horizontal spacing between nodes in the same layer

    for i, layer_size in enumerate(layer_sizes):
        layer_num = i + 1
        x_pos = i * h_spacing
        y_offset = (max(layer_sizes) - layer_size) / 2
        for j in range(layer_size):
            node_id = f'L{layer_num}_N{j+1}'
            y_pos = j + y_offset
            G.add_node(node_id)
            pos[node_id] = (x_pos, -y_pos)  # Flip y to have positive up
            node_labels[node_id] = ''  # Empty labels for nodes
            if layer_num == 1:
                colors.append('lightblue')
            elif layer_num < num_layers:
                colors.append('lightgreen')
            else:
                colors.append('lightcoral')

    for i in range(num_layers - 1):
        layer_size_a = layer_sizes[i]
        layer_size_b = layer_sizes[i + 1]
        for j in range(layer_size_a):
            for k in range(layer_size_b):
                node_a = f'L{i+1}_N{j+1}'
                node_b = f'L{i+2}_N{k+1}'
                G.add_edge(node_a, node_b)

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, labels=node_labels, node_color=colors, with_labels=True, arrows=False,
            node_size=500, font_size=10, font_color='black', edge_color='gray')
    
    plt.title("Neural Network Diagram")
    plt.show()
