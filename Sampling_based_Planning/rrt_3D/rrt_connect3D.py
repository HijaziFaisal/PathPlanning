# rrt connect algorithm
"""
This is rrt connect implementation for 3D
@author: yue qi
"""
import numpy as np
from numpy.matlib import repmat
from collections import defaultdict
import time
import matplotlib.pyplot as plt

import os
import sys

from sympy import Id

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/")

from rrt_3D.env3D import env
from rrt_3D.utils3D import getDist, sampleFree, nearest, steer, isCollide, near, visualization, cost, path, edgeset
from rrt_3D.plot_util3D import set_axes_equal, draw_block_list, draw_Spheres, draw_obb, draw_line, make_transparent


class Tree():
    def __init__(self, node):
        self.V = []
        self.Parent = {}
        self.V.append(node)
        # self.Parent[node] = None

    def add_vertex(self, node):
        if node not in self.V:
            self.V.append(node)
        
    def add_edge(self, parent, child):
        # here edge is defined a tuple of (parent, child) (qnear, qnew)
        self.Parent[child] = parent


class rrt_connect():
    def __init__(self, start, goal, config):
        self.env = env(**config)
        self.Parent = {}
        self.V = []
        self.E = set()
        self.i = 0
        self.maxiter = 10000
        # self.stepsize = 0.0005 #FIXME: use this instead
        self.stepsize = 0.5 
        self.Path = []
        self.done = False

        self.env.start = start
        self.env.goal = goal

        self.qinit = tuple(start)
        self.qgoal = tuple(goal)
        self.x0, self.xt = tuple(start), tuple(goal)
        self.qnew = None
        self.done = False
        
        self.ind = 0
        self.fig = plt.figure(figsize=(10, 8))


#----------Normal RRT algorithm
    def BUILD_RRT(self, qinit):
        tree = Tree(qinit)
        for k in range(self.maxiter):
            qrand = self.RANDOM_CONFIG()
            self.EXTEND(tree, qrand)
        return tree

    def EXTEND(self, tree, q):
        qnear = tuple(self.NEAREST_NEIGHBOR(q, tree))
        qnew, dist = steer(self, qnear, q)
        self.qnew = qnew # store qnew outside
        if self.NEW_CONFIG(q, qnear, qnew, dist=None):
            tree.add_vertex(qnew)
            tree.add_edge(qnear, qnew)
            if qnew == q:
                return 'Reached'
            else:
                return 'Advanced'
        return 'Trapped'

    def NEAREST_NEIGHBOR(self, q, tree):
        # find the nearest neighbor in the tree
        V = np.array(tree.V)
        if len(V) == 1:
            return V[0]
        xr = repmat(q, len(V), 1)
        dists = np.linalg.norm(xr - V, axis=1)
        return tuple(tree.V[np.argmin(dists)])

    def RANDOM_CONFIG(self):
        return tuple(sampleFree(self))

    def NEW_CONFIG(self, q, qnear, qnew, dist = None):
        # to check if the new configuration is valid or not by 
        # making a move is used for steer
        # check in bound
        collide, _ = isCollide(self, qnear, qnew, dist = dist)
        return not collide

#----------RRT connect algorithm
    def CONNECT(self, Tree, q):
        # print('in connect')
        while True:
            S = self.EXTEND(Tree, q)
            if S != 'Advanced':
                break
        return S

    def RRT_CONNECT_PLANNER(self, qinit, qgoal, visualize=False):
        Tree_A = Tree(qinit)
        Tree_B = Tree(qgoal)
        for k in range(self.maxiter):
            # print(k)
            qrand = self.RANDOM_CONFIG()
            if self.EXTEND(Tree_A, qrand) != 'Trapped':
                qnew = self.qnew # get qnew from outside
                if self.CONNECT(Tree_B, qnew) == 'Reached':
                    # print('reached')
                    self.done = True
                    self.Path = self.PATH(Tree_A, Tree_B)
                    if visualize:
                        self.visualization(Tree_A, Tree_B, k)
                        plt.show()
                    return
                    # return
            Tree_A, Tree_B = self.SWAP(Tree_A, Tree_B)
            if visualize:
                self.visualization(Tree_A, Tree_B, k)
        return 'Failure'

    # def PATH(self, tree_a, tree_b):
    def SWAP(self, tree_a, tree_b):
        tree_a, tree_b = tree_b, tree_a
        return tree_a, tree_b

    def PATH(self, tree_a, tree_b):
        qnew = self.qnew
        patha = []
        pathb = []
        while True:
            patha.append((tree_a.Parent[qnew], qnew))
            qnew = tree_a.Parent[qnew]
            if qnew == self.qinit or qnew == self.qgoal:
                break
        qnew = self.qnew
        while True:
            pathb.append((tree_b.Parent[qnew], qnew))
            qnew = tree_b.Parent[qnew]
            if qnew == self.qinit or qnew == self.qgoal:
                break
        return list(reversed(patha)) + pathb

#----------RRT connect algorithm        
    def visualization(self, tree_a, tree_b, index):
        if (index % 20 == 0 and index != 0) or self.done:
            # a_V = np.array(tree_a.V)
            # b_V = np.array(tree_b.V)
            Path = self.Path
            start = self.env.start
            goal = self.env.goal
            a_edges, b_edges = [], []
            for i in tree_a.Parent:
                a_edges.append([i,tree_a.Parent[i]])
            for i in tree_b.Parent:
                b_edges.append([i,tree_b.Parent[i]])
            ax = plt.subplot(111, projection='3d')
            ax.view_init(elev=90., azim=0.)
            ax.clear()
            draw_Spheres(ax, self.env.balls)
            draw_block_list(ax, self.env.blocks)
            if self.env.OBB is not None:
                draw_obb(ax, self.env.OBB)
            draw_block_list(ax, np.array([self.env.boundary]), alpha=0)
            draw_line(ax, a_edges, visibility=0.75, color='g')
            draw_line(ax, b_edges, visibility=0.75, color='y')
            draw_line(ax, Path, color='r')
            ax.plot(start[0:1], start[1:2], start[2:], 'go', markersize=7, markeredgecolor='k')
            ax.plot(goal[0:1], goal[1:2], goal[2:], 'ro', markersize=7, markeredgecolor='k')
            set_axes_equal(ax)
            make_transparent(ax)
            ax.set_axis_off()
            plt.pause(0.0001)


def flatten(coords):
    output = {'coordinates': []} 
    coordinates = []
    for i in coords:
        felement, _ = i
        longitude, latitude, altitude = felement
        # if(altitude >= 0.1): altitude = 15.0
        # else: altitude = 10.0 
        coordinates.append([longitude,latitude,altitude])
    output['coordinates'] = coordinates
    #return patha + pathb
    return coordinates

def pairwise_overlap(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a = iter(iterable)
    b = iter(iterable)
    next(b)
    return zip(a, b)

def triwise_overlap(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    assert len(iterable) > 2
    a = iter(iterable)
    b = iter(iterable)
    c = iter(iterable)
    next(b)
    next(c)
    next(c)
    return zip(a, b, c)

def remove_redundant_points(path):
    """
    mutates (modifies) the `path` variable, removes redundant points
    """
    if len(path) <= 3:
        return
    def angle(p1,p2):
        x1,y1,z1 = p1
        x2,y2,z2 = p2
        dx = x2 - x1
        dy = y2 - y1
        return np.arctan(dy/(dx + 0.00001*np.sign(dx)))
    indexes_to_remove = []
    for i,(p1,p2,p3) in enumerate(triwise_overlap(path)):
        if np.abs(angle(p1,p2) - angle(p1,p3)) <= 5:
            indexes_to_remove.append(i+1)

    for i in reversed(indexes_to_remove):
        # print(i)
        del path[i]


def find_new_path(droneID,start,goal,env_config,active_paths,visualize=False):
    """
    droneID, Integer ID number of the drone whose path we are generating
    start, (double)[x,y,z] The start point of the mission
    goal, (double)[x,y,z] The goal of our mission
    env, Dictionary containing the boundaries of the environment and all the objects inside it
    active_paths, Dictionary containing the active drone IDs and the paths they are following
    """
    #modify active paths 
    
    DRONE_EXTENTS = 0.2
    ZLAYER_HEIGHT = 0.5
    PADDING = 0.1
    env_config = deepcopy(env_config)

    for droneId, path in active_paths.items():
        for p1, p2 in pairwise_overlap(path):
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            dx = x2 - x1
            dy = y2 - y1
            px = (x1 + x2)/2
            py = (y1 + y2)/2
            pz = (z1 + z2)/2-PADDING/2
            #TODO: pathfinder output should have discrete z
            ex = np.sqrt(dx**2 + dy**2)/2
            ox = np.arctan(dy / (dx+0.00001*np.sign(dx)))
            # ox = np.arctan(dx / (dy+0.00001*np.sign(dy)))
            #TODO: implement ox and oy
            env_config['obbs'] += [[[px,py,pz], [ex+PADDING,DRONE_EXTENTS+PADDING,ZLAYER_HEIGHT+PADDING], [ox,0,0]]]

    # create rrt connect class
    p = rrt_connect(start, goal, env_config)
    p.RRT_CONNECT_PLANNER(p.qinit, p.qgoal, visualize=visualize)
    coords_flattened = flatten(p.Path)
    # print('path     :', coords_flattened)
    # coords_flattened = coords_flattened[2:-2]
    remove_redundant_points(coords_flattened)
    # print('pathundup:', coords_flattened)
    active_paths[droneID] = coords_flattened
    return coords_flattened

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_config',
                        help='path to json file containing environment objects')
    parser.add_argument('-v', '--visualize', action='store_true')
    args = parser.parse_args()

    import json
    from copy import deepcopy
    with open(args.env_config, 'r') as f:
        env_config = json.load(f)
    start = np.array([2.0, 2.0, 2.0])
    goal = np.array([6.0, 16.0, 0.0])


    active_paths = {
        # 0: [[5.862774675369044, 14.811551134482752, 1.0], [5.7252361071054585, 14.334534696199228, 1.0], [5.587697538841872, 13.857518257915702, 1.0], [5.450158970578285, 13.380501819632176, 1.0], [5.312620402314699, 12.90348538134865, 1.0], [5.175081834051113, 12.426468943065125, 1.0], [5.0375432657875265, 11.9494525047816, 1.0], [4.90000469752394, 11.472436066498075, 1.0], [4.762466129260353, 10.995419628214549, 1.0], [4.624927560996767, 10.518403189931025, 1.0], [4.48738899273318, 10.0413867516475, 1.0], [4.349850424469594, 9.564370313363977, 1.0], [4.212311856206007, 9.087353875080451, 1.0], [4.0747732879424206, 8.610337436796925, 1.0], [3.937234719678834, 8.1333209985134, 1.0], [3.7996961514152474, 7.6563045602298745, 1.0], [3.6621575831516613, 7.17928812194635, 1.0], [3.5246190148880747, 6.702271683662825, 1.0], [3.3870804466244886, 6.2252552453793, 1.0], [3.249541878360902, 5.748238807095775, 1.0], [3.1120033100973155, 5.27122236881225, 1.0], [2.9744647418337293, 4.794205930528725, 1.0], [2.836926173570143, 4.3171894922452, 1.0], [2.6993876053065566, 3.8401730539616756, 1.0], ],
    }

    starttime = time.time()
    find_new_path(1,start,goal,env_config,active_paths,visualize=args.visualize)
    # print('time used = ' + str(time.time() - starttime))
    
    # starttime = time.time()
    # find_new_path(2,start,goal,env_config,active_paths,visualize=args.visualize)
    # print('time used = ' + str(time.time() - starttime))

    # starttime = time.time()
    # find_new_path(3,start,goal,env_config,active_paths,visualize=args.visualize)
    # print('time used = ' + str(time.time() - starttime))

    # start = np.array([2.0, 2.0, 5.0])
    # goal = np.array([6.0, 16.0, 5.0])

    # starttime = time.time()
    # find_new_path(4,start,goal,env_config,active_paths,visualize=args.visualize)
    # print('time used = ' + str(time.time() - starttime))

