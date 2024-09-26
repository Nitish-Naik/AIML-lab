
def a_star_search(graph, start, goal, h):
    # Create two lists: one for visited nodes and one for nodes to explore
    open_list = [start]
    closed_list = []
    
    # Create a dictionary to store the cost of reaching each node
    g_cost = {start: 0}  # g(n): cost from start node to n
    f_cost = {start: h[start]}
    
    # Create a dictionary to store the parent node of each node
    parent = {start: None}

    while open_list:
        # Get the node from open_list with the lowest f_cost
        current_node = min(open_list, key=lambda node: f_cost[node])

        # If we have reached the goal node, reconstruct and return the path
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = parent[current_node]
            return path[::-1]  # Return reversed path
        
        # Move current node from open_list to closed_list
        open_list.remove(current_node)
        closed_list.append(current_node)

        # Explore the neighbors of the current node
        for neighbor, cost in graph[current_node]:
            if neighbor in closed_list:
                continue  # Ignore already explored neighbors
            
            # Calculate tentative g_cost for the neighbor
            tentative_g_cost = g_cost[current_node] + cost

            if neighbor not in open_list:
                open_list.append(neighbor)  # Discover a new node

            # If the new path to the neighbor is worse, skip updating
            elif tentative_g_cost >= g_cost[neighbor]:
                continue
            
            # This path is the best so far, so record it
            parent[neighbor] = current_node
            g_cost[neighbor] = tentative_g_cost
            f_cost[neighbor] = g_cost[neighbor] + h[neighbor]

    return None  # No path found


# Function to take dynamic graph input
def get_dynamic_graph():
    graph = {}
    num_nodes = int(input("Enter the number of nodes: "))
    
    for _ in range(num_nodes):
        node = input("Enter node name: ").strip()
        graph[node] = []
        num_neighbors = int(input(f"Enter the number of neighbors for node {node}: "))
        
        for _ in range(num_neighbors):
            neighbor = input(f"Enter neighbor of {node}: ").strip()
            cost = float(input(f"Enter the cost to reach {neighbor} from {node}: "))
            graph[node].append((neighbor, cost))
    
    return graph

# Function to take dynamic heuristic input
def get_dynamic_heuristic():
    heuristic = {}
    num_nodes = int(input("Enter the number of nodes for heuristic: "))
    
    for _ in range(num_nodes):
        node = input("Enter node name for heuristic: ").strip()
        h_value = float(input(f"Enter heuristic value for node {node}: "))
        heuristic[node] = h_value
    
    return heuristic

# Main program
graph = get_dynamic_graph()
heuristic = get_dynamic_heuristic()

# Taking inputs dynamically for start and goal nodes
start_node = input("Enter the start node: ").strip()
goal_node = input("Enter the goal node: ").strip()


print(graph, heuristic)
# Perform A* search
path = a_star_search(graph, start_node, goal_node, heuristic)

if path:
    print("Path found:", " -> ".join(path))
else:
    print("No path found")

"""
Enter the number of nodes: 8
Enter node name: S
Enter the number of neighbors for node S: 2
Enter neighbor of S: A
Enter the cost to reach A from S: 3
Enter neighbor of S: D
Enter the cost to reach D from S: 4
Enter node name: A
Enter the number of neighbors for node A: 2
Enter neighbor of A: B
Enter the cost to reach B from A: 4
Enter neighbor of A: D
Enter the cost to reach D from A: 5
Enter node name: D
Enter the number of neighbors for node D: 1
Enter neighbor of D: E
Enter the cost to reach E from D: 2
Enter node name: B
Enter the number of neighbors for node B: 2
Enter neighbor of B: C
Enter the cost to reach C from B: 4
Enter neighbor of B: E
5Enter the cost to reach E from B: 5
Enter node name: E
Enter the number of neighbors for node E: 1
Enter neighbor of E: F
Enter the cost to reach F from E: 4
Enter node name: F
Enter the number of neighbors for node F: 1
Enter neighbor of F: G
Enter the cost to reach G from F: 3.5
Enter node name: C
Enter the number of neighbors for node C: 1
Enter neighbor of C: B
Enter the cost to reach B from C: 4
Enter node name: G
Enter the number of neighbors for node G: 1
Enter neighbor of G: F
Enter the cost to reach F from G: 3.5
Enter the number of nodes for heuristic: 8
Enter node name for heuristic: S
Enter heuristic value for node S: 11.5
Enter node name for heuristic: A
Enter heuristic value for node A: 10.1
Enter node name for heuristic: D
Enter heuristic value for node D: 9.2
Enter node name for heuristic: B
Enter heuristic value for node B: 5.8
Enter node name for heuristic: E
Enter heuristic value for node E: 7.1
Enter node name for heuristic: C
Enter heuristic value for node C: 3.4
Enter node name for heuristic: F
Enter heuristic value for node F: 3.5
Enter node name for heuristic: G
Enter heuristic value for node G: 0
Enter the start node: S
Enter the goal node: G
{'S': [('A', 3.0), ('D', 4.0)], 'A': [('B', 4.0), ('D', 5.0)], 'D': [('E', 2.0)], 'B': [('C', 4.0), ('E', 55.0)], 'E': [('F', 4.0)], 'F': [('G', 3.5)], 'C': [('B', 4.0)], 'G': [('F', 3.5)]} {'S': 11.5, 'A': 10.1, 'D': 9.2, 'B': 5.8, 'E': 7.1, 'C': 3.4, 'F': 3.5, 'G': 0.0}
Path found: S -> D -> E -> F -> G """

















































