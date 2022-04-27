import pickle
import time
import pandas as pd
import numpy as np
from D_Heuristics import Heuristics
import multiprocessing
from itertools import product

class Evaluation:

    def __init__(self, link_df, demand_df, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls, trans_penalty = 5):
        self.adjList = {}  # To store graph: u -> (v,w)
        self.link_s = link_s # string variable, source column name
        self.link_t = link_t # string variable, target column name
        self.link_df = link_df.astype({self.link_s:str, self.link_t:str}) # dataframe for creating road network
        self.link_w = link_w # (string) travel cost column name
        self.demand_s = demand_s # string variable
        self.demand_t = demand_t # string variable
        self.demand_w = demand_w # (string) travel demand column name
        self.demand_df = demand_df.astype({self.demand_s:str, self.demand_t:str})
        self.route_ls = route_ls
        self.trans_penalty = trans_penalty
        self.un_penalty = 50 + 24.74

    def graph_from_routes(self):
        route_id = 0
        for each_route in self.route_ls:
            # iterate each_route till the second-to-last node
            for stop_order in range(len(each_route)-1):
                stop_1, stop_2 = each_route[stop_order], each_route[stop_order+1]
                edge_cost = self.link_df[(self.link_df[self.link_s] == stop_1) & (self.link_df[self.link_t] == stop_2)][self.link_w].values[0]
                self._add_edge(stop_1, stop_2, edge_cost, route_id)

            route_id += 1

    def _make_twin_node(self, node, second_node, edge_weight, route):
            node_twin = str(route) + "_" + str(node)
            self.adjList[node_twin] = [(second_node, edge_weight, route)]
            self.adjList[v] = [(u_twin, w, r)]
            self.adjList[u_twin].append((u, self.trans_penalty, r))
            self.adjList[u].append((u_twin, self.trans_penalty, r))

    def _add_edge(self, u, v, w, r):
        #  Edge going from node u to v and v to u with weight w
        # u (w)-> v, v (w) -> u
        # Check if u already in graph and if it's new u from another transit route
        if (u in self.adjList.keys()) and (v in self.adjList.keys()) and (r not in [each[2] for each in self.adjList[u]]) and (r not in [each[2] for each in self.adjList[v]]):
            u_twin = str(r) + "_" + str(u)
        if (u in self.adjList.keys()) and (r not in [each[2] for each in self.adjList[u]]) :
            u_twin = str(r) + "_" + str(u)
            self.adjList[u_twin] = [(v, w, r)]
            self.adjList[v] = [(u_twin, w, r)]
            self.adjList[u_twin].append((u, self.trans_penalty, r))
            self.adjList[u].append((u_twin, self.trans_penalty, r))
            
        elif (u in self.adjList.keys()) and (v in self.adjList.keys()):
            self.adjList[u].append((v, w, r))
            self.adjList[v].append((u, w, r))
        elif (u in self.adjList.keys()) and (v not in self.adjList.keys()):
            self.adjList[u].append((v, w, r))
            self.adjList[v] = [(u, w, r)]
        elif (u not in self.adjList.keys()) and (v in self.adjList.keys()):
            self.adjList[u] = [(v, w, r)]
            self.adjList[v].append((u, w, r))
        else:
            self.adjList[u] = [(v, w, r)]
            self.adjList[v] = [(u, w, r)]

    
    def djikstra(self, s, e):
        vis = {key: False for key in self.adjList.keys()}
        prev = {key: None for key in self.adjList.keys()}
        dist = {key: None for key in self.adjList.keys()}  # init distance table
        passed_routes = {key: None for key in self.adjList.keys()} # store routes passed by a shortest path
        dist[s] = 0
        pq_key = np.array([s]) #init priority queue
        pq_value = np.array([0])

        while len(pq_key) > 0:
            index_pos, minValue = pq_value.argmin(), pq_value.min()
            index = pq_key[index_pos]
            vis[index] = True
            pq_value = np.delete(pq_value, index_pos)
            pq_key = np.delete(pq_key, index_pos)

            # if a smaller (compared to the min in pq) value already exists in dist, skip current iteration
            if dist[index] < minValue: continue
            for each_tuple in self.adjList[index]:
                each_neighbor = each_tuple[0]
                if vis[each_neighbor]: continue
                newDist = dist[index] + each_tuple[1]
                if (dist[each_neighbor] == None) or (newDist < dist[each_neighbor]):
                    prev[each_neighbor] = index
                    dist[each_neighbor] = newDist
                    passed_routes[each_neighbor] = each_tuple[2] # add route_id to the dict of tuples
                    pq_key = np.append(pq_key, each_neighbor)
                    pq_value = np.append(pq_value, newDist)
            
            if index == e:
                break
        return (dist[e], prev, passed_routes)

    def findShortestPath(self, s, e):
        path = []
        routes = []
        # find lists of virtual (twin) stops fo nodes s and e
        s_vir_ls = [each[0] for each in self.adjList[s] if each[0].endswith('_'+s)]
        e_vir_ls = [each[0] for each in self.adjList[e] if each[0].endswith('_'+e)]
        s_vir_ls.append(s)
        e_vir_ls.append(e)
        # a list of unpacked results, including distance(dist), prev(previous stop list) and a list of routes bus stops belonged to.
        unpacked_ls = []
        alter_ODpair_ls = [] # all feasible paths
        for each_ODpair in list(product(s_vir_ls, e_vir_ls)):
            each_unpacked = self.djikstra(each_ODpair[0], each_ODpair[1])
            if each_unpacked[0] == None: continue
            unpacked_ls.append(each_unpacked)
            alter_ODpair_ls.append(each_ODpair)
        
        if len(unpacked_ls) > 0:
            chosen_pair = np.argmin([each[0] for each in unpacked_ls])
            dist, prev, passed_routes = unpacked_ls[chosen_pair]
            s, e = alter_ODpair_ls[chosen_pair]
        else: return path

        at = e
        while at != None:
            path.append(at)
            routes.append(passed_routes[at])
            at = prev[at]
        path.reverse()
        routes.reverse()
        return (dist[e], path, routes)

def multi_eval(link_df, demand_df, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls):

    cores = multiprocessing.cpu_count() - 2

    demand_df = Heuristics.to_undirected_df(demand_df, source=demand_s, target=demand_t)
    df_od_split = np.array_split(demand_df, cores)
    params = []
    for i in range(len(df_od_split)):
        params.append([link_df, df_od_split[i], link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls])

    with multiprocessing.Pool(cores) as pool:
        fitness_ls, num_d0_ls, num_d1_ls, num_d2_ls, num_dun_ls = zip(*pool.map(fitness, params))
        pool.close()
        pool.join()

def single_eval(link_df, demand_df, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls):
    
    demand_df = Heuristics.to_undirected_df(demand_df, source=demand_s, target=demand_t)

    params = [link_df, demand_df, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls]

    fitness(params)
        

def fitness(params):
    
    total_travel_cost = 0
    num_d0 = 0
    num_d1 = 0
    num_d2 = 0
    num_dun = 0
        
    link_df, demand_df, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls = params
    eval = Evaluation(link_df, demand_df, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls)
    eval.graph_from_routes()

    for each in eval.demand_df[[eval.demand_s,eval.demand_t,eval.demand_w]].values:
        unpacked = eval.findShortestPath(each[0], each[1])
        if len(unpacked) == 0:
            num_dun += each[2]
            continue
        unit_cost, path, routes = unpacked

        np.unique(routes)
        
        
        
if __name__ == '__main__':
    start = time.time()

    df_links = pd.read_csv('./data/mumford3_links.txt')
    df_demand = pd.read_csv('./data/mumford3_demand.txt')
    file_name = 'init_route_sets.pkl'
    open_file = open(file_name, "rb")
    route_ls = pickle.load(open_file)
    open_file.close()

    link_s, link_t, link_w, demand_s, demand_t, demand_w = 'from', 'to', 'travel_time', 'from', 'to', 'demand'

    # multi_eval(df_links, df_demand, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls)
    single_eval(df_links, df_demand, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls)
    
