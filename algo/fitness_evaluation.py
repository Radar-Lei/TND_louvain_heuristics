import pickle
import time
import pandas as pd
import numpy as np
from D_Heuristics import Heuristics
import multiprocessing
from collections import Counter
from itertools import chain, product
from numpy.lib.stride_tricks import sliding_window_view
import re

class Evaluation:

    def __init__(self, link_df, demand_df, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls):
        self.adjList = None  # To store graph: u -> (v,w)
        self.link_s = link_s # string variable, source column name
        self.link_t = link_t # string variable, target column name
        self.link_df = link_df.astype({self.link_s:str, self.link_t:str}) # dataframe for creating road network
        self.link_w = link_w # (string) travel cost column name
        self.demand_s = demand_s # string variable
        self.demand_t = demand_t # string variable
        self.demand_w = demand_w # (string) travel demand column name
        self.demand_df = demand_df.astype({self.demand_s:str, self.demand_t:str})
        self.route_ls = route_ls
        self.nodes_in_routes = None
        self.trans_penalty = 5
        # self.un_penalty = 50 + 24.74

    def graph_from_routes(self):
        route_id = 0
        adjList_matrix = []
        for each_route in self.route_ls:
            adjList = {}
            # iterate each_route till the second-to-last node
            for stop_order in range(len(each_route)-1):
                stop_1, stop_2 = each_route[stop_order], each_route[stop_order+1]
                edge_cost = self.link_df[(self.link_df[self.link_s] == stop_1) & (self.link_df[self.link_t] == stop_2)][self.link_w].values[0]
                if stop_1 in adjList.keys():
                    adjList[stop_1].append([stop_2, edge_cost, route_id])
                else:
                    adjList[stop_1] = [[stop_2, edge_cost, route_id]]
                
                adjList[stop_2] = [[stop_1, edge_cost, route_id]]

                # adjList[stop_2] = [(stop_1, edge_cost, route_id)]
                # self._add_edge(stop_1, stop_2, edge_cost, route_id)
            adjList_matrix.append(adjList)
            route_id += 1
        self._transfer_linker(adjList_matrix)
        print('#nodes:{}'.format(len(self.adjList.keys())))

    def _transfer_linker(self, adjlist_matrix):
        # a list of tuples denoting repeated stops (transfer stop)
        trans_counter = Counter(chain.from_iterable([each.keys() for each in adjlist_matrix])).most_common()
        
        for each_oldkey in [each[0] for each in trans_counter]:
            # a list to stop ranamed stops for the same transfer stop, e.g., '0_1', '3_1', ...
            renamed_node_ls = []
            for each_route_adj in adjlist_matrix:
                if each_oldkey in each_route_adj.keys():
                    # create new_key for the transfer stop with format " route_oldkey"
                    new_key = str(each_route_adj[each_oldkey][0][2]) + '_' + each_oldkey
                    renamed_node_ls.append(new_key)
                    # update key of the node itself
                    each_route_adj[new_key] = each_route_adj.pop(each_oldkey)
                    
                    # update values (oldkey) in the node's neighbors
                    for each_edge in each_route_adj[new_key]:
                        each_neighbor = each_edge[0]
                        each_neighbors_neighbors = each_route_adj[each_neighbor]
                        each_route_adj[each_neighbor] = [[new_key, each[1], each[2]] if each[0] == each_oldkey else each for each in each_neighbors_neighbors]

                    # connect transfer nodes 
                    for each in renamed_node_ls:
                        # find route contain the transfer node
                        if (each.endswith('_'+ each_oldkey)) and (each != new_key):
                            tmp_route_adj = adjlist_matrix[int(re.split('_',each)[0])]
                            tmp_route_adj[each].append([new_key, self.trans_penalty, 'transfer'])
                            each_route_adj[new_key].append([each, self.trans_penalty, 'transfer'])
        # merge the list of dicts into one dict
        self.adjList = {k: v for d in adjlist_matrix for k, v in d.items()}

    def djikstra(self, s, e):
        vis = {key: False for key in self.adjList.keys()}
        prev = {key: None for key in self.adjList.keys()}
        dist = {key: None for key in self.adjList.keys()}  # init distance table
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
                    pq_key = np.append(pq_key, each_neighbor)
                    pq_value = np.append(pq_value, newDist)
            if index == e:
                break
        return (dist, prev)

    def _path_transfer_counter(self, path):
        """
        count the number of transfers in a path
        """
        transfer_counter = 0
        for each_pair in sliding_window_view(np.array(path), window_shape=2):
            if [each[2] for each in self.adjList[each_pair[0]] if each[0] == each_pair[1]][0] == 'transfer':
                transfer_counter += 1
            
        return transfer_counter

    def _flatten(self, potential_list): 
        new_list = [] 
        for e in potential_list: 
            if isinstance(e, list): 
                new_list.extend(self._flatten(e)) 
            else: 
                new_list.append(e) 
        return new_list

    def findShortestPath(self, s, e):
        
        if (s not in self.nodes_in_routes) or (e not in self.nodes_in_routes): return []
        # find lists of source and target nodes formated as "route_stop"
        source_ls = [each_new_key for each_new_key in self.adjList.keys() if each_new_key.endswith('_'+s)]
        target_ls = [each_new_key for each_new_key in self.adjList.keys() if each_new_key.endswith('_'+e)]
        
        od_pair_ls = list(product(source_ls, target_ls))

        last_fitness = float('inf')
        last_path = []
        # last_transfer_num = float('inf')
        for each_od_pair in od_pair_ls:
            path = []
            o, d = each_od_pair
            dist, prev = self.djikstra(o, d)
            if dist[d] == None: continue
            at = d
            while at != None:
                path.append(at)
                at = prev[at]
            path.reverse()
            num_transfer = self._path_transfer_counter(path)
            # dist[d] == last_fitness ---> break the tie if identical fitness but less transfers
            if dist[d] < last_fitness or (dist[d] == last_fitness and num_transfer < last_transfer_num):
                last_fitness = dist[d]
                last_transfer_num = num_transfer
                last_path = path

        if last_fitness < float('inf'):
            return (last_fitness, last_path, last_transfer_num)

        else:
            return []

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
    
    total_demand = demand_df[demand_w].sum()

    return sum(fitness_ls) / total_demand, sum(num_d0_ls) / total_demand, sum(num_d1_ls) / total_demand, sum(num_d2_ls) / total_demand, sum(num_dun_ls) / total_demand

def single_eval(link_df, demand_df, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls):
    
    demand_df = Heuristics.to_undirected_df(demand_df, source=demand_s, target=demand_t)

    params = [link_df, demand_df, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls]

    total_costs, num_d0, num_d1, num_d2, num_dun = fitness(params)
    total_demand = demand_df[demand_w].sum()
    
    return  total_costs / total_demand, num_d0 / total_demand, num_d1 / total_demand, num_d2 / total_demand, num_dun / total_demand

def fitness(params):
    trans_penalty = 5
    un_penalty = 50 + 24.74
    total_travel_cost = 0
    num_d0 = 0
    num_d1 = 0
    num_d2 = 0
    num_dun = 0
        
    link_df, demand_df, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls = params
    eval = Evaluation(link_df, demand_df, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls)
    eval.graph_from_routes()
    

    # collapse a list of lists to check if s and e in route_ls
    nodes_in_routes = np.unique(np.array(eval._flatten(route_ls)))
    
    

    eval.nodes_in_routes = nodes_in_routes

    for each in eval.demand_df[[eval.demand_s,eval.demand_t,eval.demand_w]].values:
        
        unpacked = eval.findShortestPath(each[0], each[1])
        if len(unpacked) == 0:
            num_dun += each[2]
            total_travel_cost += each[2] * un_penalty
            continue
        unit_cost, path, num_transfer = unpacked
        if num_transfer > 2:
            num_dun += each[2]
            total_travel_cost += each[2] * un_penalty
        elif num_transfer == 2:
            num_d2 += each[2]
            total_travel_cost += each[2] * unit_cost # unit_cost already contain the transfer penalty
        elif num_transfer == 1:
            num_d1 += each[2]
            total_travel_cost += each[2] * unit_cost
        else:
            num_d0 += each[2]
            total_travel_cost += each[2] * unit_cost
    
    return total_travel_cost, num_d0, num_d1, num_d2, num_dun
        
        
if __name__ == '__main__':
    start = time.time()

    df_links = pd.read_csv('./data/mumford3_links.txt')
    df_demand = pd.read_csv('./data/mumford3_demand.txt')
    file_name = 'init_route_sets_3.pkl'
    open_file = open(file_name, "rb")
    route_ls = pickle.load(open_file)
    open_file.close()
    # with open('Mumford_Mumford3_solution.txt', 'rU') as f:
    #         route_ls = []
    #         for ele in f:
    #             line = ele.split('\n')
    #             route_ls.append([str(each) for each in line[0].split(',')])

    link_s, link_t, link_w, demand_s, demand_t, demand_w = 'from', 'to', 'travel_time', 'from', 'to', 'demand'

    ATT, d0, d1, d2, dun = multi_eval(df_links, df_demand, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls)
    # ATT, d0, d1, d2, dun = single_eval(df_links, df_demand, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls)
    print('The ATT for the optimized_route_set is: {:.2f}'.format(ATT))
    print('d0: {:.2f}'.format(d0))
    print('d1: {:.2f}'.format(d1))
    print('d2: {:.2f}'.format(d2))
    print('dun: {:.2f}'.format(dun))

    print('The evaluation took %4.3f minutes' % ((time.time() - start) / 60))