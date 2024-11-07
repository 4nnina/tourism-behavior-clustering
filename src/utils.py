import osmnx as ox
import os

def check_folder():
    folder = '../output/graphs/poiway/kmedoid'
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    
    folder = '../output/graphs/poiway/kmeans'
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    folder = '../output/graphs/catway/kmedoid'
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    folder = '../output/graphs/catway/kmeans'
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    folder = '../output/graphs/levway/methodA'
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


    folder = '../output/matrix/poiway'
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    folder = '../output/matrix/catway'
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    folder = '../output/matrix/levway/methodA'
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    folder = '../output/distance_matrix'
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

def check_map():
    #load Verona map
    graph_path = '../dataset/Verona.graphml'

    if not os.path.exists(graph_path):
        graph_area = ("Verona, Italy")
        G = ox.graph_from_place(graph_area, network_type='walk')
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        ox.save_graphml(G, graph_path)