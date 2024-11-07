from data_classes import Data
from utils import check_folder, check_map

check_map()
check_folder()

dtype_dataoptions = {'activation_date': str, 'id_vc': str, 'visit_timestamp' : str, 'profile_vc': str, 'poi' : int}
dtype_infooptions = {'poi_id': int, 'poi_name': str, 'category_id' : int, 'category_name' : str, 'longitude' : float, 'latitude' : float }
data1422_path = '../dataset/data_POI_2014_to_2022.csv'
dataInfo_path = '../dataset/poi_info.csv'
                
obj = Data(data1422_path, dataInfo_path, dtype_dataoptions, dtype_infooptions, 'id_vc', 'visit_timestamp', 'poi', 'poi_id', 'poi_name', 
               'category_id', 'category_name', 'longitude', 'latitude')

obj.map_poiinformations()

# Kmeans by category
obj.transform_data_forencoding(0)

# Kmeans formula 
obj.transform_data_forencoding(1)

# Levdistance and kmedoid + final kmeans
obj.transform_data_forencoding(2)


obj.delete_files()