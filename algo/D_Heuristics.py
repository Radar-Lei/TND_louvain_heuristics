#!/usr/bin/python

""" Author: Da Lei (David) greatradar@gmail.com

    Algorithm Name: A travel-demand based heuristics for generating initial route sets

    Main Idea: Weighting edges in road network considering both distances and travel demands, giving preference to nodes which have more 
    passengers traveling to vertices that have already been included in the transit route. 
    
    Keywords: Eager dijkstra algorithm with early stop, heap optimization, travel-demand based cost function, dynamical programming
"""
import pickle
import numpy as np
import pandas as pd
import copy
import heapq

class Heuristics:
    def __init__(self, link_df, demand_df, link_s, link_t, link_w, demand_s, demand_t, demand_w, undirected=True, normalization=True):

        if len(link_df) + len(demand_df) < 2:
            print("Empty dataframe")

        self.adjList = {}  # To store graph: u -> (v,w)
        self.link_s = link_s # string variable, source column name
        self.link_t = link_t # string variable, target column name
        self.link_w = link_w # (string) travel cost column name
        self.demand_s = demand_s # string variable
        self.demand_t = demand_t # string variable
        self.demand_w = demand_w # (string) travel demand column name

        self.link_df = link_df.astype({self.link_s:str, self.link_t:str}) # dataframe for creating road network
        self.demand_df = demand_df.astype({self.demand_s:str, self.demand_t:str})        

        self.undirected = undirected # boolean variable, true for converting directed to undirected
        self.normalization = normalization

        self.norm_demand_df = demand_df.copy(deep=True)
        self.norm_link_df = link_df.copy(deep=True)

    def _add_edge(self, u, v, w):
        #  Edge going from node u to v and v to u with weight w
        # u (w)-> v, v (w) -> u
        # Check if u already in graph
        if u in self.adjList.keys():
            self.adjList[u].append((v, w))
        else:
            self.adjList[u] = [(v, w)]
    
    def _normalization(self, df, weight_col_name):
        return (df[weight_col_name] - df[weight_col_name].min()) / (df[weight_col_name].max() - df[weight_col_name].min())

    def show_graph(self):
        # u -> v(w)
        for u in self.adjList:
            print(u, "->", " -> ".join(str(f"{v}({w})") for v, w in self.adjList[u]))

    @classmethod
    def to_undirected_df(cls, original_df, source=None, target=None):
        """
        converted directed dataframe to the undirected.
        s,t
        1,2     1,2
        2,1     2,3    
        2,3 ->   
        3,2     
        """
        tmp_df = pd.DataFrame(np.sort(original_df[[source,target]], axis=1))
        return original_df[~tmp_df.duplicated()]

    def _graph_from_pd(self):
        # if self.undirected:
        #     self.link_df = self.to_undirected_df(self.link_df, source=self.link_s, target=self.link_t)
        for _, row in self.link_df.iterrows():
            self._add_edge(row[self.link_s], row[self.link_t], row[self.link_w])
        
            

    def _node_weight(self, node_id, node_in_route):
        """
        node_id: the concerned node
        node_in_route: nodes already included in the route.
        """
        # find passenger flows between node_id and node_in_route (dataframe)
        tmp_flow_df = self.norm_demand_df[((self.norm_demand_df[self.demand_s] == node_id) & (self.norm_demand_df[self.demand_t].isin(node_in_route))) | ((self.norm_demand_df[self.demand_s].isin(node_in_route)) & (self.norm_demand_df[self.demand_t] == node_id))]
        if len(tmp_flow_df) == 0:
            return 1

        return 1 / (tmp_flow_df[self.demand_w].sum() + 1)

    def _update_vis(self, vis, path):
        for each in path:
            vis[each] = True

    def _drop_return(self, df, index):
        row = df.loc[index]
        df.drop(index, inplace=True)
        return row

    def RouteSet(self, n_routes, l_min, l_max, importance):

        """
        n_routes: number of routes constraint

        l_min, l_max: number of transit stops constraint

        importance: controlling the relative importance (node weights compared to edge distance)

        """

        # construct road network
        self._graph_from_pd()

        route_set = []
        # demand_df = self.demand_df.sort_values(by=self.demand_w, ascending=False).copy(deep=True) # make sure self.demand_df unchanged
        # converted to undirected dataframe
        # demand_df = Heuristics.to_undirected_df(demand_df, source=self.demand_s, target=self.demand_t)
        # demand_df.reset_index(drop=True, inplace=True)

        while len(route_set) < n_routes:
            if self.normalization:
                self.norm_demand_df[self.demand_w] = self._normalization(self.demand_df, weight_col_name=self.demand_w)
                self.norm_link_df[self.link_w] = self._normalization(self.link_df, weight_col_name=self.link_w)

            
            last_source, last_target = self.demand_df.loc[self.demand_df[self.demand_w].idxmax(), [self.demand_s, self.demand_t]].values
            # last_source, last_target = demand_df.iloc[row_loc][[self.demand_s, self.demand_t]].values
            # customized
            vis = {key: False for key in self.adjList.keys()}
            norm_cost, one_route = self.findShortestPath(last_source, last_target, importance, copy.deepcopy(vis))
            # update visited nodes list using route
            self._update_vis(vis, one_route)

            trail_1 = 0
            while len(one_route) < l_max and (trail_1<=3):
                # find the travel demand dataframe with target node as the new source node.
                tmp_partial_demand_df = self.demand_df[self.demand_df[self.demand_s] == last_target].sort_values(by=self.demand_w, ascending=False).copy(deep=True)
                # the target of the route_segment cannot be thoes already in one_route.
                tmp_partial_demand_df =  tmp_partial_demand_df[~tmp_partial_demand_df[self.demand_t].isin(one_route)]

                # find available new target for the new source node
                trail_2 = 0
                while (len(tmp_partial_demand_df) > 0) and (trail_2<=3):
                    
                    curr_target = self._drop_return(tmp_partial_demand_df, tmp_partial_demand_df[self.demand_w].idxmax())[self.demand_t]
                    # curr_target = tmp_partial_demand_df.loc[tmp_partial_demand_df[self.demand_w].idxmax(), [self.demand_t]].values
                    # find shortest path between last target and current target and compute node weight with one_route as the ex_path
                    tmp_unpacked = self.findShortestPath(last_target, curr_target, importance, copy.deepcopy(vis), one_route, l_max)

                    if len(tmp_unpacked) == 0:
                        trail_2 += 1
                        continue

                    _, one_route_segment  = tmp_unpacked

                    if len(one_route + one_route_segment) <= l_max:
                        # append segment but remove the last_target from the route segment
                        one_route += one_route_segment
                        self._update_vis(vis, one_route_segment)
                        tmp_partial_demand_df =  tmp_partial_demand_df[~tmp_partial_demand_df[self.demand_t].isin(one_route)]
                        last_target = one_route_segment[-1]
                        break
                    trail_2 += 1

                trail_1 += 1

            if len(one_route) >= l_min:
                route_set.append(one_route)
                self._update_demand(one_route)

            if len(route_set) == 59:
                print('')

        return route_set

    def _update_demand(self, one_route):

        self.demand_df.drop(self.demand_df[self.demand_df[self.demand_s].isin(one_route) & self.demand_df[self.demand_t].isin(one_route)].index, inplace=True)
        

    def djikstra(self, s, e, importance, vis, ex_path):
        
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
                # collect nodes already in route, including start and end terminals, also including nodes in ex_path when extending the route.
                node_in_route = {s,e}.union(prev).union(ex_path)
                # filter None, need to convert to a list since filter is a generator function.
                node_in_route = list(filter(None, node_in_route))
                
                # calculate node costs of the two vertices 
                node_weights = (self._node_weight(index, node_in_route) + self._node_weight(each_neighbor, node_in_route)) / 2
                
                newDist = dist[index] + (1-importance) * each_tuple[1] + importance * node_weights
                if (dist[each_neighbor] == None) or (newDist < dist[each_neighbor]):
                    prev[each_neighbor] = index
                    dist[each_neighbor] = newDist
                    pq_key = np.append(pq_key, each_neighbor)
                    pq_value = np.append(pq_value, newDist)
            if index == e:
                break
        return (dist, prev)

    def findShortestPath(self, s, e, importance, vis, ex_path=[], l_max=25):
        """
        s - start node
        e - end node
        ex_path - path generated by the first-time djikstra, used for computing weights of nodes in augmenting route
        vis - (dict) recording nodes in ex_path, to avoid loop when augmenting route.
        """
        dist, prev = self.djikstra(s, e, importance, vis, ex_path)
        path = []
        if dist[e] == None: return path
        at = e
        while at != None:
            path.append(at)
            at = prev[at]
        path.reverse()
        diff_len = len(ex_path) + len(path) - l_max # 22, 5,25
        # diff_len - 1 since we need to remove the source of the route_seg (target of the one_route) when extending the route.
        if (len(ex_path) > 0) and (diff_len - 1) > 0:
            path = path[1:diff_len+1]
        elif len(ex_path) > 0:
            path = path[1:]

        return (dist[e], path)

if __name__ == "__main__":

    df_links = pd.read_csv('./data/mumford3_links.txt')
    df_demand = pd.read_csv('./data/mumford3_demand.txt')

    h = Heuristics(link_df=df_links, demand_df=df_demand, link_s='from', link_t='to', link_w='travel_time', demand_s='from', demand_t='to', demand_w='demand')

    route_set = h.RouteSet(n_routes=60, l_min=12, l_max=25, importance=0.5)

    file_name = 'init_route_sets.pkl'
    open_file = open(file_name, 'wb')
    pickle.dump(route_set, open_file)
    open_file.close()

    print('complete')