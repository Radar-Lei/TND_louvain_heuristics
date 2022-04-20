#!/usr/bin/python

""" Author: Radar-Lei
    lazy dijkstra algorithm implementation for the lecture ( with early stop and return both shortest distance and path)
    type of graph: weighted and directed
    Side notes: use the graph instance shown in the topsort lecture (without the negative edge)
"""
import numpy as np


class Graph:
    def __init__(self):
        self.adjList = {}  # To store graph: u -> (v,w)

    def add_edge(self, u, v, w):
        #  Edge going from node u to v and v to u with weight w
        # u (w)-> v, v (w) -> u
        # Check if u already in graph
        if u in self.adjList.keys():
            self.adjList[u].append((v, w))
        else:
            self.adjList[u] = [(v, w)]

    def show_graph(self):
        # u -> v(w)
        for u in self.adjList:
            print(u, "->", " -> ".join(str(f"{v}({w})") for v, w in self.adjList[u]))

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

    def findShortestPath(self, s, e):
        dist, prev = self.djikstra(s, e)
        path = []
        if dist[e] == None: return path
        at = e
        while at != None:
            path.append(at)
            at = prev[at]
        path.reverse()
        return (dist[e], path)

if __name__ == "__main__":
    g = Graph()  # False for SSSP on DAG, True for SSLP on DAG
    g.add_edge('A', 'B', 3)
    g.add_edge('A', 'C', 6)
    g.add_edge('B', 'C', 4)
    g.add_edge('B', 'D', 4)
    g.add_edge('B', 'E', 11)
    g.add_edge('C', 'D', 8)
    g.add_edge('C', 'G', 11)
    g.add_edge('D', 'E', 4)
    g.add_edge('D', 'F', 5)
    g.add_edge('D', 'G', 2)
    g.add_edge('E', 'H', 9)
    g.add_edge('F', 'H', 1)
    g.add_edge('G', 'H', 2)
    g.add_edge('H', 'H', 0)
    g.show_graph()

    dist, path = g.findShortestPath('A','H')
    print('Shortest distance:')
    print(dist)
    print('Shortest path:')
    print(path)