import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch, Patch

def visualize_process_with_legend(V, E, capacity, Pd, xp):
    # Create a directed graph
    G = nx.DiGraph()
    G.add_nodes_from(V)

    # Add edges to the graph
    for i, edge in enumerate(E):
        u, v = edge
        G.add_edge(u, v, capacity=capacity[i])

    # Position nodes using a layout
    pos = nx.spring_layout(G, seed=42)

    # Compute flow values for each path (columns of Pd)
    flows_per_edge = Pd * xp  # Each column represents the flow of a path
    total_flow = np.sum(flows_per_edge, axis=1)  # Sum flow contributions across all paths
    final_obj_value = sum(total_flow)  # Total flow or objective value

    # Update the graph with total flow values
    for i, edge in enumerate(E):
        G[edge[0]][edge[1]]['flow'] = total_flow[i]

    # Define distinct colors for paths
    num_paths = Pd.shape[1]
    cmap = plt.get_cmap("tab10")
    path_colors = [cmap(i) for i in range(num_paths)]

    # Plot the base graph (nodes only)
    plt.figure(figsize=(8, 7))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=700,
        node_color='lightgreen',
        font_size=12,
        font_weight='bold',
    )


    # Draw edges with multiple colors using FancyArrowPatch
    ax = plt.gca()
    for i, (u, v) in enumerate(E):
        if G.has_edge(u, v):
            # Get edge position
            x_start, y_start = pos[u]
            x_end, y_end = pos[v]
            dx, dy = x_end - x_start, y_end - y_start

            # print("！！！！！！！！！！！！：", num_paths)
            # Draw multiple bands for each edge
            for path_idx in range(num_paths):
                # print(f"xp[{path_idx}]: ", xp[path_idx])
                if xp[path_idx] == 0:
                   if Pd[i, path_idx] > 0:
                      color = path_colors[path_idx]
                      offset = 0.05 * (path_idx - num_paths / 2)  # Offset for each path
                      curve_rad = 0.2 + 0.05 * path_idx  # Adjust curvature for separation
                      arrow = FancyArrowPatch(
                          (x_start, y_start),
                          (x_end, y_end),
                          connectionstyle=f"arc3,rad={curve_rad}",
                          arrowstyle="-|>",
                          linewidth=3,  # Arrow width
                          linestyle=(0, (5, 2)),  # Dashed line: (0, (dash_length, gap_length))
                          color=color,
                          alpha=0.8,
                          mutation_scale=15,  # Arrowhead size
                          zorder=2
                      )
                      ax.add_patch(arrow)
                      continue

                if Pd[i, path_idx] > 0 :  # Only draw for non-zero flow
                    color = path_colors[path_idx]
                    offset = 0.05 * (path_idx - num_paths / 2)  # Offset for each path
                    curve_rad = 0.2 + 0.05 * path_idx  # Adjust curvature for separation
                    arrow = FancyArrowPatch(
                        (x_start, y_start),
                        (x_end, y_end),
                        connectionstyle=f"arc3,rad={curve_rad}",
                        arrowstyle="-|>",
                        linewidth=3,  # Arrow width
                        color=color,
                        alpha=0.8,
                        mutation_scale=15,  # Arrowhead size
                        zorder=2
                    )
                    ax.add_patch(arrow)




    # Add flow/capacity labels to edges
    edge_labels = {
        (u, v): f"{G[u][v]['flow']:.0f}/{G[u][v]['capacity']}"
        for u, v in G.edges()
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_color="black",
        font_size=10,
    )

    # Create legend for paths
    legend_handles = [
        Patch(color=path_colors[path_idx], label=f"Path {path_idx + 1}: Flow {xp[path_idx]:.2f}")
        for path_idx in range(num_paths)
    ]
    plt.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.95, 0.95),
        title="Path Flows",
        fontsize=10,
    )

    plt.title(f"Graph with flows (Sum of flows: {np.sum(xp):.2f})",
              fontsize=16,
              x=0.5,  # Horizontal position (0: far left, 1: far right)
              y=0.02  # Vertical position (default is ~1 for above the plot)
    )
    plt.tight_layout()
    plt.show()
