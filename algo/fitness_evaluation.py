import pickle
import time
import pandas as pd
import numpy as np
from D_Heuristics import Heuristics
import multiprocessing
from collections import Counter
from itertools import chain
from numpy.lib.stride_tricks import sliding_window_view



class Evaluation:

    def __init__(self, link_df, demand_df, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls):
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
        self.nodes_in_routes = None
        # self.trans_penalty = trans_penalty
        # self.un_penalty = 50 + 24.74

    def graph_from_routes(self):
        route_id = 0
        for each_route in self.route_ls:
            # iterate each_route till the second-to-last node
            for stop_order in range(len(each_route)-1):
                stop_1, stop_2 = each_route[stop_order], each_route[stop_order+1]
                edge_cost = self.link_df[(self.link_df[self.link_s] == stop_1) & (self.link_df[self.link_t] == stop_2)][self.link_w].values[0]
                self._add_edge(stop_1, stop_2, edge_cost, route_id)

            route_id += 1

    def _add_edge(self, u, v, w, r):
        #  Edge going from node u to v and v to u with weight w
        # u (w)-> v, v (w) -> u
        # Check if u already in graph
        if u in self.adjList.keys():
            self.adjList[u].append((v, w, r))
            # self.adjList[v].append((u, w, r))
        else:
            self.adjList[u] = [(v, w, r)]

        if v in self.adjList.keys():
            self.adjList[v].append((u, w, r))
            # self.adjList[v].append((u, w, r))
        else:
            self.adjList[v] = [(u, w, r)]

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

    def _same_route_counter(self, path):
        """
        count the largest possible number of repeated routes in a path
        """
        alter_route_ls = []
        for each_pair in sliding_window_view(np.array(path), window_shape=2):
            alter_route_ls.append([each[2] for each in self.adjList[each_pair[0]] if each[0] == each_pair[1]])
        
        counter = Counter(chain.from_iterable(alter_route_ls))

        return counter.most_common(1)[0][1] # return the largest number of repeated routes

    def _flatten(self, potential_list): 
        new_list = [] 
        for e in potential_list: 
            if isinstance(e, list): 
                new_list.extend(self._flatten(e)) 
            else: 
                new_list.append(e) 
        return new_list

    def findShortestPath(self, s, e):
        path = []
        if (s not in self.nodes_in_routes) or (e not in self.nodes_in_routes): return path
        dist, prev = self.djikstra(s, e)
        if dist[e] == None: return path
        at = e
        while at != None:
            path.append(at)
            at = prev[at]
        path.reverse()
        num_transfer = len(path) - self._same_route_counter(path) - 1
        return (dist[e], path, num_transfer)      

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

    # collapse a list of lists to cgheck if s and e in route_ls
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
            total_travel_cost += each[2] * (unit_cost + 2*trans_penalty)
        elif num_transfer == 1:
            num_d1 += each[2]
            total_travel_cost += each[2] * (unit_cost + trans_penalty)
        else:
            num_d0 += each[2]
            total_travel_cost += each[2] * unit_cost
    
    return total_travel_cost, num_d0, num_d1, num_d2, num_dun
        
        
if __name__ == '__main__':
    start = time.time()

    df_links = pd.read_csv('./data/mumford3_links.txt')
    df_demand = pd.read_csv('./data/mumford3_demand.txt')
    file_name = 'init_route_sets.pkl'
    open_file = open(file_name, "rb")
    route_ls = pickle.load(open_file)
    open_file.close()
    # with open('Islam_Mumford3_solution.txt', 'rU') as f:
    #         route_ls = []
    #         for ele in f:
    #             line = ele.split('\n')
    #             route_ls.append([str(each) for each in line[0].split(',')])

    link_s, link_t, link_w, demand_s, demand_t, demand_w = 'from', 'to', 'travel_time', 'from', 'to', 'demand'

    # ATT, d0, d1, d2, dun = multi_eval(df_links, df_demand, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls)
    ATT, d0, d1, d2, dun = single_eval(df_links, df_demand, link_s, link_t, link_w, demand_s, demand_t, demand_w, route_ls)
    print('The ATT for the optimized_route_set is: {}'.format(ATT))
    print('d0: {}'.format(d0))
    print('d1: {}'.format(d1))
    print('d2: {}'.format(d2))
    print('dun: {}'.format(dun))

    print('The evaluation took %4.3f minutes' % ((time.time() - start) / 60))