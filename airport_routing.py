import math
import requests
import heapq

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def fetch_airport_geometry(lat, lon, radius=5000):
    query = f"""
    [out:json][timeout:15];
    (
      way["aeroway"="taxiway"](around:{radius},{lat},{lon});
      way["aeroway"="runway"](around:{radius},{lat},{lon});
    );
    out geom;
    """
    
    url = "https://overpass-api.de/api/interpreter"
    try:
        response = requests.post(url, data={'data': query}, timeout=15)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching geometry: {e}")
    return None

class AirportGraph:
    def __init__(self):
        self.nodes = {}  # id -> {lat, lon}
        self.edges = {}  # id -> list of {to, distance, ref}

    def add_node(self, node_id, lat, lon):
        self.nodes[node_id] = {'lat': lat, 'lon': lon}
        if node_id not in self.edges:
            self.edges[node_id] = []

    def add_edge(self, u, v, distance, ref):
        self.edges[u].append({'to': v, 'distance': distance, 'ref': ref})
        self.edges[v].append({'to': u, 'distance': distance, 'ref': ref}) # Undirected

def build_graph(osm_data):
    graph = AirportGraph()
    runway_nodes = {} # ref -> list of node ids

    if not osm_data or 'elements' not in osm_data:
        return graph, runway_nodes

    for element in osm_data['elements']:
        if element['type'] != 'way': continue
        
        tags = element.get('tags', {})
        aeroway = tags.get('aeroway', '')
        ref = tags.get('ref', '')
        
        # Only care about taxiways with names or runways
        if aeroway == 'taxiway' and (not ref or len(ref) > 3):
            # We might still need unnamed taxiways for connectivity, label them 'Unnamed'
            ref = 'Unnamed'

        geom = element.get('geometry', [])
        if not geom: continue
        
        for i in range(len(geom)):
            node_id = f"{geom[i]['lat']}_{geom[i]['lon']}"
            graph.add_node(node_id, geom[i]['lat'], geom[i]['lon'])
            
            if aeroway == 'runway' and ref:
                # Add to runway nodes
                runways = ref.split('/')
                for rw in runways:
                    rw = rw.strip()
                    if rw not in runway_nodes:
                        runway_nodes[rw] = []
                    runway_nodes[rw].append(node_id)
            
            if i > 0:
                prev_id = f"{geom[i-1]['lat']}_{geom[i-1]['lon']}"
                dist = haversine(geom[i-1]['lat'], geom[i-1]['lon'], geom[i]['lat'], geom[i]['lon'])
                graph.add_edge(prev_id, node_id, dist, ref if aeroway == 'taxiway' else 'Runway')

    return graph, runway_nodes

def find_nearest_node(graph, lat, lon):
    min_dist = float('inf')
    best_node = None
    for node_id, data in graph.nodes.items():
        dist = haversine(lat, lon, data['lat'], data['lon'])
        if dist < min_dist:
            min_dist = dist
            best_node = node_id
    return best_node

def find_shortest_path(graph, start_node, end_nodes):
    queue = [(0, start_node, [])]
    visited = set()
    
    while queue:
        dist, current, path = heapq.heappop(queue)
        
        if current in visited: continue
        visited.add(current)
        
        if current in end_nodes:
            return path
            
        for edge in graph.edges.get(current, []):
            if edge['to'] not in visited:
                heapq.heappush(queue, (dist + edge['distance'], edge['to'], path + [edge['ref']]))
                
    return None

def get_taxi_route(lat, lon, active_runway):
    print("🗺️ Building topological map for taxi routing...")
    data = fetch_airport_geometry(lat, lon)
    graph, runway_nodes = build_graph(data)
    
    if not graph.nodes or not active_runway:
        return []
        
    start_node = find_nearest_node(graph, lat, lon)
    
    # Try to find nodes for the selected runway
    end_nodes = []
    # e.g., active_runway = "18"
    for rw_key in runway_nodes:
        if active_runway in rw_key:
            end_nodes.extend(runway_nodes[rw_key])
            
    # Fallback to nearest runway node if specific runway not found properly parsed
    if not end_nodes:
        for rw_key in runway_nodes:
             end_nodes.extend(runway_nodes[rw_key])
             
    if not start_node or not end_nodes:
        return []
        
    raw_path = find_shortest_path(graph, start_node, set(end_nodes))
    if not raw_path:
        return []
        
    # Condense consecutive identical instructions
    condensed_route = []
    for step in raw_path:
        if step not in ('Unnamed', 'Runway') and (not condensed_route or condensed_route[-1] != step):
            condensed_route.append(step)
            
    return condensed_route
