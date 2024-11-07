import os
import warnings
import multiprocessing as mp 
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import time
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, pairwise_distances_argmin
from joblib import Parallel, delayed
from Levenshtein import distance
import folium
import h5py

class Data:

    def __init__(self,datapath,infopath,dtype_dataoptions, dtype_infooptions,
                id_vc_col, timestamp_col, poi_col,
                poi_id_col, poi_name_col, category_id_col, category_name_col,longitude_col,latitude_col
                ):
        self.data = pd.read_csv(datapath, dtype=dtype_dataoptions)
        self.info = pd.read_csv(infopath, dtype = dtype_infooptions)
        
        
        #info column
        self.poi_id_col = poi_id_col
        self.poi_name_col = poi_name_col
        self.category_id_col = category_id_col
        self.category_name_col = category_name_col
        self.longitude_col = longitude_col
        self.latitude_col = latitude_col

        #data info
        self.id_vc_col = id_vc_col
        self.timestamp_col = timestamp_col
        self.poi_col = poi_col


        self.distance_cache = {}
        self.poi_map = {}
        
        self.G = ox.load_graphml("../dataset/Verona.graphml")

        self.poiway_result = {}
        self.poiway_matrix = {}
        self.best_k_foryear_poiway = {}
        self.best_value_foryear_poiway = {}
        
        self.catway_result = {}
        self.catway_matrix = {}
        self.best_k_foryear_catway = {}
        self.best_value_foryear_catway = {}

        self.levdistance_matrix = {}
        self.levway_matrix = {}
        self.levway_result = {}
        #best in kmeans
        self.best_k_foryear_kmeans = {}
        self.best_value_foryear_kmeans= {}
        #best in kmedoid
        self.best_k_foryear_kmedoid= {}
        self.best_value_foryear_kmedoid= {}
        self.centroidIndex_methodA = {}

        self.methodB_matrix = {}
        self.methodB_result = {}
        #best in kmeans
        self.best_k_foryear_methodB_ = {}
        self.best_value_foryear_methodB_= {}
        self.centroidIndex_methodB = {}

        #id_veronacard	profilo	data_visita	ora_visita	sito_nome	poi

    def map_poiinformations(self):
        # Create a dictionary to map POIs to their respective attributes
        for index, row in self.info.iterrows():
            # Initialize poi_id before using it
            poi_id = row[self.poi_id_col]
            
            # Create a dictionary with initial poi information
            poi_info = {
                'poi_name': row[self.poi_name_col],
                'category_id': [],
                'category_name': [],
                'longitude': row[self.longitude_col],
                'latitude': row[self.latitude_col]
            }
            
            # Store the poi_info in the poi_map using poi_id as the key
            self.poi_map[poi_id] = poi_info

        # Map category_id and category_name to the correct POI
        for index, row in self.info.iterrows():
            poi_id = row[self.poi_id_col]
            self.poi_map[poi_id]['category_id'].append(row[self.category_id_col])
            self.poi_map[poi_id]['category_name'].append(row[self.category_name_col])
    
    def get_coordinates(self,poi_id):
      if poi_id in self.poi_map:
        poi_info = self.poi_map[poi_id]
        return poi_info[self.longitude_col], poi_info[self.latitude_col]
      else:
        print('ERROR (get coordinates): poi is not mapped')
        return None, None

    def get_category(self,poi_id):
      if poi_id in self.poi_map:
        poi_info = self.poi_map[poi_id]
        return poi_info[self.category_id_col]
      else:
        print(f'ERROR (get category): poi {poi_id} is not mapped')
        return None 

    def getDistance(self,lon_dest, lat_dest, lon_origin, lat_origin):
        origin_node = ox.nearest_nodes(self.G, lon_origin, lat_origin)
        destination_node = ox.nearest_nodes(self.G, lon_dest, lat_dest)
    
        distance_in_meters = nx.shortest_path_length(self.G, origin_node, destination_node, weight='length')
        return distance_in_meters 
        
    # Get distance between two POIs
    def get_distance_cached(self, poi_id_origin, poi_id_dest):
        if (poi_id_origin, poi_id_dest) in self.distance_cache:
            return self.distance_cache[(poi_id_origin, poi_id_dest)]
        else:
            # Calculate distance using the getDistance method
            lon_origin, lat_origin = self.get_coordinates(poi_id_origin)
            lon_dest, lat_dest = self.get_coordinates(poi_id_dest)
            if lon_origin is None or lon_dest is None:
                return None
            distance = self.getDistance(lon_dest, lat_dest, lon_origin, lat_origin)
            # Save distance in the cache
            self.distance_cache[(poi_id_origin, poi_id_dest)] = distance
            return distance

    def calculate_time_slot(self,timestamp):
        if timestamp.time() < time(12,0):
            return 1  # Morning
        elif timestamp.time() < time(14,0):
            return 2  # Noon
        elif timestamp.time() < time(18,0):
            return 3  # Afternoon
        else:
            return 4  # Evening

    # Category name * 1000 + poi, based on POIs' number
    def transform_value_poiway(self,poi):
      if(poi!=9000):
        #some POI has more than one category, the last is selected
        return self.get_category(poi)[-1] * 1000 + poi
      else:
        return 9000;
          
    # For each POI get its category
    def transform_value_onlycatway(self,poi):
      if(poi!=9000):
        # some POI has more than one category, the last is selected
        return self.get_category(poi)[-1] 
      else:
        return 10;

    def calculate_silhouette_kmedoid(self, distance_matrix, k):
        # K-Medoids with precomputed distance matrix
        kmedoids = KMedoids(n_clusters=k, metric='precomputed', init='k-medoids++', random_state=1000)
        labels = kmedoids.fit_predict(distance_matrix)
        silhouette_avg = silhouette_score(distance_matrix, labels, metric='precomputed')
        return silhouette_avg
          
    def calculate_silhouette_kmeans(self,data, k):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=1000)
        labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, labels)
        return silhouette_avg

    def find_optimal_clusters(self,data, year, encoding_type, matrix_type,method= 'A', max_clusters=20, n_jobs=2):
        """
            param 
            coding type:
            0 = kmeans
            1 = kmedoid
            
            matrix type:
            0 = poiway
            1 = catway
            2 = levway
        """
        matrix_string = ["poiway", "catway", "levway"]
        encoding_string = ["kmeans" , "kmedoid"]
        K = range(2, max_clusters + 1)
    
        # Silhoette computation
        if(encoding_type==0):
            silhouette_scores = Parallel(n_jobs=n_jobs)(delayed(self.calculate_silhouette_kmeans)(data, k) for k in K)
        else:
           silhouette_scores = Parallel(n_jobs=n_jobs)(delayed(self.calculate_silhouette_kmedoid)(data, k) for k in K) 
        score_copy = silhouette_scores.copy()
            
        best_value = 1
        
        while best_value>= 0.99:
            optimal_k = K[np.argmax(score_copy)]   
            best_value = round(max(score_copy),2)
            score_copy[np.argmax(score_copy)] = -1

        plt.figure(figsize=(10, 6))
        plt.plot(K, silhouette_scores, 'bo-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score') 
        plt.title(f'Silhouette {matrix_string[matrix_type]} {encoding_string[encoding_type]} For Optimal k')
        plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
        plt.legend()
        plt.grid(True)
        plt.show

        if matrix_type != 2:
            plt.savefig(f'../output/graphs/{matrix_string[matrix_type]}/{encoding_string[encoding_type]}/silhouettescore_{year}.png')
        else:
            plt.savefig(f'../output/graphs/{matrix_string[matrix_type]}/method{method}/{encoding_string[encoding_type]}/silhouettescore_{year}.png')
    
        return optimal_k, best_value
    
    def levenshtein_distance(self, stringa1, stringa2):
        return distance(stringa1, stringa2)
    
    def compute_distance( self, lock,  final_dataset , start_index, end_index, col_number, year ):
        # --> I create the submatrix from the matrix <--
        row_number = end_index - start_index
        matrix_distance = np.zeros(shape=(row_number, col_number), dtype=np.uint16)
        for i in range(start_index, end_index):
            # --> Difference between this item and the others <--
            string_one = final_dataset['stringa'].iloc[i]

            for j in range(i+1, col_number):
                # --> I extract the j-th row <--
                string_two = final_dataset['stringa'].iloc[j]
                distance_sum = 0
                distance_sum = self.levenshtein_distance(string_one , string_two ) 
                matrix_distance[ i - start_index , j ] = distance_sum
                             
        # Save the matrix
        dataset_name = str(start_index) + '_' + str(end_index)
        with lock:
            with h5py.File(f'../output/distance_matrix/matrix_year_{year}.hdf5', 'a') as f:
                    dset = f.create_dataset(f"{dataset_name}", 
                                            data = matrix_distance,
                                            compression='gzip',
                                            dtype=np.uint16
                                           )
                   
    def save_zipfile(self,file_path,dataset_name, matrix):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset(dataset_name, data=matrix, compression='gzip', dtype=np.uint16)

    def get_shortest_path(self,lon_dest, lat_dest, lon_origin, lat_origin):
        origin_node = ox.nearest_nodes(self.G, lon_origin, lat_origin)
        destination_node = ox.nearest_nodes(self.G, lon_dest, lat_dest)
        path = nx.shortest_path(self.G, origin_node, destination_node, weight='length')
        return path

    def delete_files(self):
        directory = 'distance_matrix'

        if os.path.isdir(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"File {file_path} deleted.")
        else:
            print(f"Directory {directory} does not exist.")
            def get_transparency(self,count, max_count):
                if max_count == 0:
                    return 0.5
                return 0.1 + 0.9 * (count / max_count)
            
    def process_cluster(self,cluster_label, centroids):
        cluster_data = centroids[centroids['clusters_label'] == cluster_label]
    
        # Verona is the center of the map
        cluster_map = folium.Map(location=[45.4384, 10.9916], zoom_start=13)
    
        path_cache = {}
        path_counts = {}
    
        first_viewed = set()
        last_viewed = set()
        all_viewed = set()
    
        colors = sns.color_palette("hsv", len(centroids['clusters_label'].unique())).as_hex()
    
        # Data processing
        for _, row in cluster_data.iterrows():
            path = []
            
            for i in range(15):  # Max 15 POI columns
                attr = row.get(f'POI{i}', 9000)
                if attr == 9000:
                    break

                if attr in self.poi_map:
                    poi_info = self.poi_map[poi]
                    path.append((poi_info['latitude'], poi_info['longitude']))
                    all_viewed.add(poi)
                else:
                    print(f"POI {poi} does not found in poi_map")
    
            if len(path) < 2:
                continue
    
            for j in range(len(path) - 1):
                start = path[j]
                end = path[j + 1]
                segment = (start, end)
                if segment not in path_cache:
                    path_cache[segment] = self.get_shortest_path(end[1], end[0], start[1], start[0])
    
                path_nodes = path_cache[segment]
                path_coords = [(self.G.nodes[node]['y'], self.G.nodes[node]['x']) for node in path_nodes]
    
                if segment not in path_counts:
                    path_counts[segment] = 0
                path_counts[segment] += 1

        max_count = max(path_counts.values()) if path_counts else 1
    
        # Plot paths on the map
        for segment, count in path_counts.items():
            transparency = self.get_transparency(count, max_count)
            path_nodes = path_cache[segment]
            path_coords = [(self.G.nodes[node]['y'], self.G.nodes[node]['x']) for node in path_nodes]
            folium.PolyLine(path_coords, color=colors[cluster_label], weight=3.5, opacity=transparency).add_to(cluster_map)
    
        first_viewed = set()
        last_viewed = set()
        all_viewed = set()
        
        for _, row in cluster_data.iterrows():
            for i in range(15):  # Max 15 POI columns
                poi = row.get(f'POI{i}', 9000)
                
                if poi == 9000:
                    break
                    
                if poi in self.poi_map:
                    if i == 0:
                        first_viewed.add(poi)
                    if i == len(row) - 1 or row.get(f'POI{i + 1}', 9000) == 9000:
                        last_viewed.add(poi)
                    all_viewed.add(poi)
    
        # Add marker in the map
        for poi_id, poi_info in self.poi_map.items():
            colore = 'gray'
            if poi_id in all_viewed:
                if poi_id in first_viewed and poi_id in last_viewed:
                    colore = 'orange'  # First and last
                elif poi_id in first_viewed:
                    colore = 'blue'  # Only first
                elif poi_id in last_viewed:
                    colore = 'red'  # Only last
                else:
                    colore = 'green' # seen at least once
    
            folium.Marker(
                location=[poi_info['latitude'], poi_info['longitude']],
                popup=f"POI {poi_id}: {poi_info['poi_name']}",
                icon=folium.Icon(color=colore)
            ).add_to(cluster_map)
            
        legend_html = '''
        <div style="position: fixed;
            padding: 25px;
            border-radius: 25px;
            bottom: 50px; 
            left: 50px; 
            width: 290px; 
            height: 250px; 
            background-color: #f9f9f9; 
            z-index: 9999; 
            font-size: 14px; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
            color: #333;">
            <b style="display: block; margin-bottom: 12px; font-size: 16px; color: #222;">Legenda marker:</b>
            
            <div style="margin-bottom: 8px; line-height: 1.5;">
                <i class="fa fa-map-marker fa-2x" style="color: orange; vertical-align: middle;"></i>&nbsp;Sia per primo che per ultimo
            </div>
            <div style="margin-bottom: 8px; line-height: 1.5;">
                <i class="fa fa-map-marker fa-2x" style="color: blue; vertical-align: middle;"></i>&nbsp;Per primo
            </div>
            <div style="margin-bottom: 8px; line-height: 1.5;">
                <i class="fa fa-map-marker fa-2x" style="color: red; vertical-align: middle;"></i>&nbsp;Per ultimo
            </div>
            <div style="margin-bottom: 8px; line-height: 1.5;">
                <i class="fa fa-map-marker fa-2x" style="color: green; vertical-align: middle;"></i>&nbsp;Nè primo nè ultimo
            </div>
            <div style="line-height: 1.5;">
                <i class="fa fa-map-marker fa-2x" style="color: gray; vertical-align: middle;"></i>&nbsp;Mai Visualizzato
            </div>lass="fa fa-map-marker fa-2x" style="color: gray; vertical-align: middle;"></i>&nbsp;Mai Visualizzato
            </div>
        </div>
        '''
        cluster_map.get_root().html.add_child(folium.Element(legend_html))
        legend_html2 = '''
         <div style="position: fixed;
         padding: 20px;
         border-radius: 25px;
         bottom: 50px; right: 50px; width: 200px; height: 180px; 
         background-color: white; z-index:9999; font-size:14px;">
         <b>Legenda percorsi:</b><br>
         <i class="fa fa-map-marker fa-2x" style="color:gray"></i>&nbsp;Todo<br>
         </div>
         '''
        #cluster_map.get_root().html.add_child(folium.Element(legend_html2))
  
        cluster_map.save(f'cluster_map_{cluster_label}.html')

    def process_cluster2(self,clusters, centroids, tracks, year,durate_somma, durate_min, durate_max,num_tracce, flag_method = 'A'):

        # Verona is the center of the map
        cluster_map = folium.Map(location=[45.4384, 10.9916], zoom_start=13)
    
        path_cache = {}
        path_counts = {}
        path_type = {}

        first_viewed = set()
        last_viewed = set()
        all_viewed = set()
        
        colors = sns.color_palette("hsv", len(centroids['clusters_label'].unique())).as_hex()
        for cluster_label in clusters:
            cluster_data = centroids[centroids['clusters_label'] == cluster_label]    
        
            # Data processing
            for _, row in cluster_data.iterrows():
                path = []
                
                for i in range(15):  # Max 15 POI columns
                    poi = row.get(f'POI{i}', 9000)
                    if poi == 9000:
                        break
        
                    if poi in self.poi_map:
                        poi_info = self.poi_map[poi]
                        path.append((poi_info['latitude'], poi_info['longitude']))
                        all_viewed.add(poi)
                    else:
                        print(f"POI {poi} does not found in poi_map")
        
                if len(path) < 2:
                    continue
        
                for j in range(len(path) - 1):
                    start = path[j]
                    end = path[j + 1]
                    segment = (start, end)
                    if segment not in path_cache:
                        path_cache[segment] = self.get_shortest_path(end[1], end[0], start[1], start[0])
        
                    path_nodes = path_cache[segment]
                    path_coords = [(self.G.nodes[node]['y'], self.G.nodes[node]['x']) for node in path_nodes]
        
                    if segment not in path_type:
                        path_type[segment] = []
                    path_type[segment].append(cluster_label)
            
        # Plot paths on the map
        for segment, type_list in path_type.items():
            transparency = 1
            path_nodes = path_cache[segment]
            path_coords = [(self.G.nodes[node]['y'], self.G.nodes[node]['x']) for node in path_nodes]
            for _, type in enumerate(type_list):
                offset_coords = [(lat, lon) for lat, lon in path_coords]
        
                folium.PolyLine(
                    offset_coords, 
                    color=colors[type], 
                    weight=5, 
                    opacity=transparency, 
                ).add_to(cluster_map)
        
        for cluster_label in clusters:
            
            cluster_data = tracks[tracks['clusters_label'] == cluster_label]
            for _, row in cluster_data.iterrows():
                for i in range(15):  # Max 15 POI columns
                    poi = row.get(f'POI{i}', 9000)
                    
                    if poi == 9000:
                        break
                        
                    if poi in self.poi_map:
                        if i == 0:
                            first_viewed.add(poi)
                        if i == len(row) - 1 or row.get(f'POI{i + 1}', 9000) == 9000:
                            last_viewed.add(poi)
                        all_viewed.add(poi)

        first_viewed_centroids = set()
        last_viewed_centroids = set()
        all_viewed_centroids = set()
        for cluster_label in clusters:
            
            cluster_data = centroids [centroids['clusters_label'] == cluster_label]
            for _, row in cluster_data.iterrows():
                for i in range(15):  # Max 15 POI columns
                    poi = row.get(f'POI{i}', 9000)
                    
                    if poi == 9000:
                        break
                        
                    if poi in self.poi_map:
                        if i == 0:
                            first_viewed_centroids.add(poi)
                        if i == len(row) - 1 or row.get(f'POI{i + 1}', 9000) == 9000:
                            last_viewed_centroids.add(poi)
                        all_viewed_centroids.add(poi)

        
        # Add marker in the map
        for poi_id, poi_info in self.poi_map.items():
            colore = 'gray'
            if poi_id in all_viewed:
                colore = 'green'
            if poi_id in all_viewed_centroids:
                colore = 'darkblue'
            
            folium.Marker(
                location=[poi_info['latitude'], poi_info['longitude']],
                popup=f"POI {poi_id}: {poi_info['poi_name']}",
                icon=folium.Icon(color=colore)
            ).add_to(cluster_map)
                
        legend_html = '''
        <div style="position: fixed;
            padding: 10px;
            border-radius: 10px;
            bottom: 50px; 
            left: 50px; 
            width: 250px; 
            height: 150px; 
            background-color: #f9f9f9; 
            z-index: 9999; 
            font-size: 12px; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
            color: #333;">

            <b style="display: block; margin-bottom: 8px; font-size: 14px; color: #222;">Legenda marker:</b>
            
            <div style="margin-bottom: 6px; line-height: 1.3;">
                <i class="fa fa-map-marker fa-lg" style="color: green; vertical-align: middle;"></i>&nbsp;Visualizzati da tracce appartenenti al centroide
            </div>
            <div style="margin-bottom: 6px; line-height: 1.3;">
                <i class="fa fa-map-marker fa-lg" style="color: blue; vertical-align: middle;"></i>&nbsp;Visualizzati solo dalla traccia del centroide
            </div>
            <div style="line-height: 1.3;">
                <i class="fa fa-map-marker fa-lg" style="color: gray; vertical-align: middle;"></i>&nbsp;Mai visualizzati
            </div>
        </div>

         '''
        cluster_map.get_root().html.add_child(folium.Element(legend_html))
        
        legend_html2 = f'''
         <div style="position: fixed;
         padding: 20px;
         border-radius: 25px;
         bottom: 50px; right: 50px; width: 250px; height: 180px; 
         background-color: white; z-index:9999; font-size:14px; line-height: 1.3;">
         <b style="margin-bottom: 3px; display: block;">Legenda centroide {clusters[0]}:</b><br>
         <span style="margin-bottom: 2px; display: block;">Durata centroide: {durate_somma} mezze ore</span>
         <span style="margin-bottom: 2px; display: block;">Durata minima: {durate_min} mezze ore</span>
         <span style="margin-bottom: 2px; display: block;">Durata massima: {durate_max} mezze ore</span>
         <span style="display: block;">Tracce totali: {num_tracce}</span>
         
         </div>
         '''
        cluster_map.get_root().html.add_child(folium.Element(legend_html2))
        
        if flag_method == 'A':
            cluster_map.save(f'../output/graphs/levway/method{flag_method}/kmeans/cluster{clusters[0]}_map_{year}.html')
        else:
            cluster_map.save(f'../output/graphs/levway/method{flag_method}/kmedoid/cluster{clusters[0]}_map_{year}.html')
        cluster_map
       
    def transform_data_forencoding(self,transform_way):
        self.data[self.timestamp_col] = pd.to_datetime(self.data[self.timestamp_col])
        maxPoiVisited = max(self.data.groupby(self.id_vc_col)[self.poi_col].apply(list).apply(len))
        
        year = pd.to_datetime(self.data[self.timestamp_col]).dt.year
        min_year = year.min()
        max_year = year.max()
        warnings.filterwarnings("ignore")
        for year in range(min_year,max_year+1):
            data_thisyear = self.data[self.data[self.timestamp_col].dt.year == year]           
            if data_thisyear.empty:
                continue
      
            data_thisyear = data_thisyear.sort_values(by=self.timestamp_col)
            
            grouped = data_thisyear.groupby(self.id_vc_col, as_index=False).apply(
                lambda x: pd.Series({
                    self.poi_col: x[self.poi_col].tolist(),
                    self.timestamp_col: x[self.timestamp_col].tolist()
                })
            ).reset_index(drop=True)
                                                                                
                
            for index,row in grouped.iterrows():
                
                first_timestamp = row[self.timestamp_col][0]
                last_timestamp = row[self.timestamp_col][-1]
                duration = int((last_timestamp - first_timestamp).total_seconds()/ 60)
                duration = np.uint16(round(duration/30))
                time_slot = np.uint8(self.calculate_time_slot(first_timestamp))
                total_distance = 0
                for i in range(len(row[self.poi_col]) - 1):
                    poi_id_origin = row[self.poi_col][i]
                    poi_id_dest = row[self.poi_col][i + 1]
                    total_distance += self.get_distance_cached(poi_id_origin, poi_id_dest)
                    
                total_distance = round(total_distance / 50) * 50  
                total_distance = np.float16(total_distance/1000)
                row[self.poi_col] += [9000] * (maxPoiVisited-len(row[self.poi_col]))

                grouped.at[index, 'duration'] = duration
                grouped.at[index, 'time_slot'] = time_slot
                grouped.at[index, 'length'] = total_distance
            
            grouped['duration'] = grouped['duration'].astype('uint16')         
            grouped['time_slot'] = grouped['time_slot'].astype('uint8')    
            grouped['length'] = grouped['length'].astype('float16')   
            grouped.drop(columns=self.timestamp_col, inplace = True)

            expanded_poi = grouped[self.poi_col].apply(pd.Series)
            expanded_poi = expanded_poi.astype('uint16')
            expanded_poi.columns = [f'POI{i}' for i in range(maxPoiVisited)]
            
            result = pd.concat([grouped, expanded_poi], axis = 1)
            result.drop(columns = [self.poi_col], inplace = True)

            poi_columns = [f'POI{i}' for i in range(maxPoiVisited)]
            desired_order = ['id_vc'] + poi_columns + ['duration','time_slot','length']
            result = result[desired_order]

            kmeans_matrix =result.copy()
           
            i =0
            if transform_way == 0:
                for col in poi_columns:
                    kmeans_matrix[col] = kmeans_matrix[col].apply(self.transform_value_poiway)

                kmeans_matrix= kmeans_matrix.drop(self.id_vc_col, axis=1).values
                
                print(f"{year} calculating silhoette")
                best_k, best_value = self.find_optimal_clusters(kmeans_matrix, year, 0, 0)
                print(f"{year} finish silhoette")
                              
                kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init='auto', random_state=1000)
                clusters = kmeans.fit_predict(kmeans_matrix)
                result['clusters_label'] = clusters
                
                self.poiway_result[year] = result.copy()
                self.poiway_matrix[year] = kmeans_matrix.copy()
                self.best_k_foryear_poiway[year] = best_k
                self.best_value_foryear_poiway[year] = best_value
                
                result.to_csv(f"../output/matrix/poiway/kmeansResult_{year}.csv", index=False) 
                if(year == max_year+1):
                    self.best_k_foryear_poiway.to_csv(f"../output/matrix/poiway/bestClusterNumber", index = False)
                    self.best_value_foryear_poiway.to_csv(f"../output/matrix/poiway/bestValue", index = False)
                
            elif transform_way == 1:
                for col in poi_columns:
                    kmeans_matrix[col] = kmeans_matrix[col].apply(self.transform_value_onlycatway)

                kmeans_matrix= kmeans_matrix.drop(self.id_vc_col, axis=1).values
                
                print(f"{year} calculating silhoette")
                best_k, best_value = self.find_optimal_clusters(kmeans_matrix, year, 0, 1)
                print(f"{year} finish silhoette")    
                kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init='auto', random_state=1000)
                clusters = kmeans.fit_predict(kmeans_matrix)
                result['clusters_label'] = clusters

                self.catway_result[year] = result.copy()
                self.catway_matrix[year] = kmeans_matrix.copy()
                self.best_k_foryear_catway[year] = best_k
                self.best_value_foryear_catway[year] = best_value

                result.to_csv(f"../output/matrix/catway/kmeansMatrix_{year}.csv", index=False) 
                if(year == max_year+1):
                    self.best_k_foryear_catway.to_csv(f"../output/matrix/catway/bestClusterNumber", index = False)
                    self.best_value_foryear_catway.to_csv(f"../output/matrix/catway/bestValue", index = False)
                       
            elif transform_way == 2:      
                
                lev_distance_strings =[]
                string_alreadyseen = {}
                for index, row in kmeans_matrix.iterrows():
                    finalstring = ""
                    for col in poi_columns:

                        catstring = "0000000000000000000000000"
                        if int(row[col]) == 9000:
                            unionstring = "#########################"
                        else:
                            
                            cat = self.get_category(int(row[col]))
                            
                            for el in cat:
                                el = el-1
                                # category 3 is missing
                                if el<=2:
                            
                                    catstring = catstring[:(el)*5] + '11111' + catstring[(el+1)*5:]
                                else:
                                    catstring = catstring[:(el-1)*5] + '11111' + catstring[(el)*5:]
                            poistring = str(row[col]).zfill(3)
                            unionstring = catstring + poistring

                    
                        finalstring = finalstring + unionstring
            
                    if finalstring in string_alreadyseen:
                        string_alreadyseen[finalstring].append(index)
                    else:
                        string_alreadyseen[finalstring] = [index]
                        
                    n = len(string_alreadyseen[finalstring])

                    if n<=1:
                        lev_distance_strings.append(finalstring)

                # print((f"Len: {len(lev_distance_strings)} \n" ))
                
                directory = '../output/distance_matrix/'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                
                final_dataset = pd.DataFrame(lev_distance_strings, columns=["stringa"])
                final_dataset.to_csv("../output/final_dataset.csv",index=False)
                
                n = final_dataset.shape[0]
                lock = mp.Lock()
                threads = 20

                step = n // threads

                process_ = []

                for thread in range(threads):
                    row_index_start = step*thread
                    if (thread+1) == threads:
                        row_index_end = n
                    else: 
                        row_index_end = row_index_start+step
                    p = mp.Process(target = self.compute_distance, args = (
                        lock,final_dataset,row_index_start,row_index_end,
                        n, year))

                    p.start()
                    process_.append(p) 
                    
                for processo in process_:
                    processo.join()

                file_path = f'../output/distance_matrix/matrix_year_{year}.hdf5'
                
                matrix = []
                
                with h5py.File(file_path, 'r') as f:
                    dataset_names = list(f.keys())
                    
                    for name in dataset_names:
                        start_index, end_index = map(int, name.split('_'))
                        
                        m = f[name][:]
                        matrix.append((start_index, end_index, m))
                        #print(f"start : {start_index}, end: {end_index}, matrix : {m}")
                
                matrix.sort(key=lambda x: x[0])
                
                final_matrix = np.concatenate([m[2] for m in matrix], axis=0)
                traspose_matrix = final_matrix.T
                
                self.levdistance_matrix[year] = final_matrix + traspose_matrix

                best_k_kmedoid, best_value_kmedoid = self.find_optimal_clusters(self.levdistance_matrix.get(year), year, 1, 2)

                #TODO silhouet
                kmedoids = KMedoids(n_clusters = best_k_kmedoid, metric = 'precomputed', random_state = 1000)
                clusters = kmedoids.fit_predict(self.levdistance_matrix.get(year))
                #kmeans_matrix = np.insert(kmeans_matrix, 1, clusters, axis=1)
                
                kmeans_matrix= kmeans_matrix.drop(self.id_vc_col, axis=1)
                for col in poi_columns:
                    kmeans_matrix = kmeans_matrix.drop([col], axis=1)

                nrighe = kmeans_matrix.shape[0]
                zeri = np.zeros(nrighe)
                kmeans_matrix = np.insert(kmeans_matrix, 0, zeri, axis=1)
                
                kmeans_matrix.astype(np.float16)
  
                j = 0
                for _,index in string_alreadyseen.items():
                    for ind in index: 
                        kmeans_matrix[ind][0] = clusters[j]
                    j = j+1
                     
                print(f"{year} calculating silhoette")
                best_k, best_value = self.find_optimal_clusters(kmeans_matrix, year, 0,2)
                print(f"{year} finish silhoette")
        
                kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init='auto', random_state=1000)
                clusters = kmeans.fit_predict(kmeans_matrix)  
                   
                result['clusters_label'] = clusters
         
                centroids = kmeans.cluster_centers_ 
                closest_indices = pairwise_distances_argmin(centroids, kmeans_matrix)
                closest_tuples = result.iloc[closest_indices]
                
                durate_min = []
                durate_max = []
                num_tracce = []
                for i in range(len(list(set(clusters)))):
                    dati = result[result['clusters_label']==i]
                    durata_min = min(dati['duration'])
                    durata_max = max(dati['duration'])
                    n = len(dati)
                    durate_min.append(durata_min)
                    durate_max.append(durata_max)
                    num_tracce.append(n)

                print(f' min: {durate_min} max: {durate_max} num_tracce{num_tracce}')
                
                for index in closest_indices:
                    cluster = result.iloc[index]['clusters_label']
                    durata = result.iloc[index]['duration']
                    self.process_cluster2([cluster], result.iloc[closest_indices], result, year,durata, durate_min[cluster], durate_max[cluster],num_tracce[cluster], flag_method= 'A')
                
                self.centroidIndex_methodA[year] = closest_indices.copy()
                self.levway_result[year] = result.copy()
                self.levway_matrix[year] = kmeans_matrix.copy()
                self.best_k_foryear_kmeans[year] = best_k
                self.best_value_foryear_kmeans  = best_value_kmedoid
                self.best_k_foryear_kmedoid[year] = best_k_kmedoid
                self.best_value_foryear_kmedoid[year] = best_value_kmedoid

                self.save_zipfile(f'../output/matrix/levway/kmedoidMatrix_{year}.hdf5', 'levdistance_matrix',self.levdistance_matrix.get(year))
                result.to_csv(f"../output/matrix/levway/methodA/kmeansResult_{year}.csv", index=False)
                np.savetxt(f"../output/matrix/levway/methodA/kmeansMatrix_{year}.csv", kmeans_matrix, delimiter=',', fmt='%f')
                
                if(year == max_year):
                    df_best_k_foryear_kmeans = pd.DataFrame.from_dict(self.best_k_foryear_kmeans, orient='index') 
                    df_best_value_foryear_kmeans = pd.Series(self.best_value_foryear_kmeans).to_frame()       
                    df_best_k_foryear_kmedoid = pd.DataFrame.from_dict(self.best_k_foryear_kmedoid, orient='index') 
                    df_best_value_foryear_kmedoid = pd.Series(self.best_value_foryear_kmedoid).to_frame()
                    df_centroidIndex_methodA = pd.DataFrame.from_dict(self.centroidIndex_methodA, orient='index') 
                    df_centroidIndex_methodA.to_csv(f"../output/matrix/levway/methodA/centroidIndex.csv", index =True)
                    df_best_k_foryear_kmeans.to_csv(f"../output/matrix/levway/methodA/bestClusterNumberKmeans.csv", index =True)
                    df_best_value_foryear_kmeans.to_csv(f"../output/matrix/levway/methodA/bestValueKmeans.csv", index =True)
                    df_best_k_foryear_kmedoid.to_csv(f"../output/matrix/levway/methodA/bestClusterNumberKmedoid.csv", index =True)
                    df_best_value_foryear_kmedoid.to_csv(f"../output/matrix/levway/methodA/bestValueNumberKmedoid.csv", index =True)
