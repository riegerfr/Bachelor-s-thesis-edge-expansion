import random
import itertools
import math
import numpy as np
# import cvxopt
# import cvxpy
import scipy
import scipy.optimize as minimize
import pickle

# todo: different ways to define 'random' hypergraphs! discuss in thesis.
# a) create vertices and add random edges until connected (given r_min and r_max)
# b) given "density":=avg #edges/node (=degree?) (needs construction?)
# c) add nodes+ one edge containing the node after another --> deg(early nodes) >> deg(late nodes)
# d) how to make random degree + random r? which distribution?


# n: # vertices
# m: # edges
# r: # max degree in edges
from connection_component import ConnectionComponent


# def update_edge(self, edge):
#     new_vertices = set()
#     new_edges = set()
#     for vertex in edge:
#         new_vertices.update(vertex.connection_component.vertices)
#         new_edges.update(vertex.edges)
#
# def update_vertex(self, vertex):
#     self.vertices.add(vertex)
#     self.edges.update(vertex.edges)
#
# def merge_with_component(self, other_component):
#     self.edges.update(other_component.edges)
#     self.vertices.update(other_component.vertices)
from edge import Edge
from poisson_process import Poisson_Process
from vertex import Vertex
from vertex_vector import Vertex_Vector


class Graph:

    def __init__(self, rank=None, max_edge_size=None, min_edge_size=None, degree=None, min_weight=None,
                 max_weight=None ):

        self.vertices = set()
        self.edges = set()  # todo: how to assure this is not a multiset?
        self.connection_components = set()

        if min_edge_size and min_edge_size < 2:
            raise ValueError('edges need to contain at least two nodes')

        if min_edge_size:  # todo: this for other arguments
            self.min_edge_size = min_edge_size
        if max_edge_size:
            self.max_edge_size = max_edge_size

        self.total_vertex_weight = 0

    def create_random_uniform_regular_graph_until_connected(self, number_vertices, rank, degree, min_weight,
                                                            max_weight):  # degree >= 2, random weights, uniform distribution. TODO: other distributions
        self.number_vertices = number_vertices
        # create vertices
        for _ in range(number_vertices):
            new_vertex = Vertex(self)
            self.vertices.add(new_vertex)
        #     self.connection_components.add(new_vertex.connection_component)

        if not (1. * self.number_vertices * degree / rank).is_integer():  # number_vertices * degree = number_edges * rank
            raise ValueError('no %s -uniform %s -regular connected hypergraph with &s vertices exists', rank, degree,
                             self.number_vertices)
        if not (degree >= 2 and rank >= 2):
            raise ValueError('rank and degree have to be at least 2')

        assert len(self.vertices) == self.number_vertices and not self.edges

        vertices_with_degree = [set() for i in range(degree + 1)]  # for ranks  0, 1, ..., rank

        vertices_with_degree[0] = self.vertices.copy()

        lowest_degree = 0
        while lowest_degree < degree:
            while len(vertices_with_degree[lowest_degree]) >= rank:
                next_edge_vertices = set(random.sample(vertices_with_degree[lowest_degree], rank))
                next_edge = Edge(self, random.uniform(min_weight, max_weight), next_edge_vertices)
                self.edges.add(next_edge)
                vertices_with_degree[lowest_degree].difference_update((next_edge_vertices))
                vertices_with_degree[lowest_degree + 1].update(next_edge_vertices)
            assert lowest_degree - 2 <= degree or not vertices_with_degree[lowest_degree]
            self.fill_up_lowest_degree(lowest_degree, max_weight, min_weight, rank, vertices_with_degree)
            lowest_degree += 1

        self.compute_connection_components()
        while len(self.connection_components) > 1:
            self.shuffle()
            self.compute_connection_components()

    def create_random_graph_by_randomly_adding_edges(self, number_vertices, rank, avg_degree, min_weight,
                                                     max_weight):
        self.number_vertices = number_vertices
        # create vertices
        for _ in range(number_vertices):
            new_vertex = Vertex(self)
            self.vertices.add(new_vertex)

        number_edges = number_vertices * avg_degree / rank  # todo: math correct? insert assert
        if not number_edges.is_integer():
            print("no integer number of edges, rounding up")
            number_edges = math.ceil(number_edges)

        for _ in range(number_edges):
            next_edge_vertices = set(random.sample(self.vertices, rank))
            for edge in self.edges:  # todo: activate
                assert not edge in next_edge_vertices  # this ensures that the two sets are not the same making the new edge actually new

            next_edge = Edge(self, random.uniform(min_weight, max_weight), next_edge_vertices)
            self.edges.add(next_edge)

        self.compute_connection_components()
        if len(self.connection_components) != 1:
            print("the created graph is not connected")

        for vertex in self.vertices:
            if vertex.degree != avg_degree:
                print(" graph not uniform")

    def compute_connection_components(self):  # todo: change to recompute?
        for vertex in self.vertices:
            vertex.connection_component = ConnectionComponent({vertex})

        for edge in self.edges:
            components = [vertex.connection_component for vertex in edge.vertices]

            vertices = set([vertex for component in components for vertex in component.vertices])
            new_component = ConnectionComponent(vertices)

            for vertex in vertices:
                vertex.connection_component = new_component


        connection_components = set()

        for vertex in self.vertices:
            connection_components.add(vertex.connection_component)


        self.connection_components = connection_components

    # def recompute_connection_components(self):
    #     connection_components_so_far = []
    #
    #     for vertex in self.vertices:
    #         vertex.connection_component = {vertex}
    #     for edge in self.edges:
    #         components_of_vertices = [vertex.connection_component for vertex in edge.vertices]
    #
    #         vertices = set([vertex for component in components_of_vertices for vertex in component])
    #
    #         for component in connection_components_so_far:
    #             if not vertices.isdisjoint(component):
    #                 vertices |= component
    #                 connection_components_so_far.remove(component)
    #
    #         for vertex in vertices:
    #             vertex.connection_component = vertices
    #
    #         connection_components_so_far.append(vertices)
    #
    #     self.connection_components = components_of_vertices

    def create_random_edges_connecting_in_the_end(self, number_vertices, rank, degree, min_weight,
                                                  max_weight):

        self.number_vertices = number_vertices
        # create vertices
        for _ in range(number_vertices):
            new_vertex = Vertex(self)
            self.vertices.add(new_vertex)

        number_edges = number_vertices * degree / rank  # todo: math correct? insert assert
        if not number_edges.is_integer():
            print("no integer number of edges, rounding up")
            number_edges = math.ceil(number_edges)

        for _ in range(number_edges):
            next_edge_vertices = set(random.sample(self.vertices, rank))
            for edge in self.edges:  # todo: activate
                assert not edge in next_edge_vertices  # this ensures that the two sets are not the same making the new edge actually new

            next_edge = Edge(self, random.uniform(min_weight, max_weight), next_edge_vertices)
            self.edges.add(next_edge)

            self.compute_connection_components()
            if len(self.connection_components) > 1:
                if len(self.connection_components) != rank:
                    for component in self.connection_components:
                        number_free_connections = 0
                        for vertex in component:
                            assert degree >= vertex.degree
                            number_free_connections += degree - vertex.degree
                        assert number_free_connections >= 1

        if len(self.connection_components) != 1:
            print("the created graph is not connected")

    def shuffle(self):
        # select two different edges

        edges = random.sample(self.edges, 2)
        vertex_0 = random.sample(edges[0].vertices, 1)[0]
        vertex_1 = random.sample(edges[1].vertices, 1)[0]

        if vertex_0.connection_component != vertex_1.connection_component:
            vertex_0.edges.remove(edges[0])
            edges[0].vertices.remove(vertex_0)
            vertex_1.edges.remove(edges[1])
            edges[1].vertices.remove(vertex_1)

            edges[0].vertices.add(vertex_0)
            edges[1].vertices.add(vertex_1)

            vertex_0.recompute()
            vertex_1.recompute()
            edges[0].recompute()
            edges[1].recompute()

    def create_random_uniform_regular_connected_graph(self, number_vertices, rank, degree, min_weight,
                                                      max_weight):  # degree >= 2, random weights, uniform distribution. TODO: other distributions
        self.number_vertices = number_vertices
        # create vertices
        for _ in range(number_vertices):
            new_vertex = Vertex(self)
            self.vertices.add(new_vertex)
        #     self.connection_components.add(new_vertex.connection_component)

        if not (
                1. * self.number_vertices * degree / rank).is_integer():  # number_vertices * degree = number_edges * rank
            raise ValueError('no %s -uniform %s -regular connected hypergraph with &s vertices exists', rank, degree,
                             self.number_vertices)
        if not (degree >= 2 and rank >= 2):
            raise ValueError('rank and degree have to be at least 2')

        assert len(self.vertices) == self.number_vertices and not self.edges

        vertices_with_degree = [set() for i in range(degree + 1)]  # for ranks  0, 1, ..., rank

        vertices_with_degree[0] = self.vertices.copy()

        # create first edge of spanning tree
        first_edge_vertices = set(random.sample(vertices_with_degree[0], rank))
        first_edge = Edge(self, random.uniform(min_weight, max_weight), first_edge_vertices)
        self.edges.add(first_edge)

        vertices_with_degree[0].difference_update((first_edge_vertices))
        vertices_with_degree[1].update(first_edge_vertices)

        # create spanning tree
        while len(vertices_with_degree[0]) >= rank - 1:
            connected_vertex = random.sample(vertices_with_degree[1], 1)[
                0]  # more efficient than random.choice (needs set to be converted to list)
            nonconnected_vertices = set(random.sample(vertices_with_degree[0], rank - 1))

            next_edge_vertices = nonconnected_vertices.copy()
            next_edge_vertices.add(connected_vertex)
            next_edge = Edge(self, random.uniform(min_weight, max_weight), next_edge_vertices)
            self.edges.add(next_edge)

            vertices_with_degree[0].difference_update((nonconnected_vertices))
            vertices_with_degree[1].update(nonconnected_vertices)

            vertices_with_degree[1].remove(connected_vertex)
            vertices_with_degree[2].add(connected_vertex)

        # in case some vertices are 'left' over, add an edge between them and those which are already connected
        self.fill_up_lowest_degree(0, max_weight, min_weight, rank, vertices_with_degree)

        lowest_degree = 1
        while lowest_degree < degree:
            while len(vertices_with_degree[lowest_degree]) >= rank:
                next_edge_vertices = set(random.sample(vertices_with_degree[lowest_degree], rank))
                next_edge = Edge(self, random.uniform(min_weight, max_weight), next_edge_vertices)
                self.edges.add(next_edge)
                vertices_with_degree[lowest_degree].difference_update((next_edge_vertices))
                vertices_with_degree[lowest_degree + 1].update(next_edge_vertices)
            assert lowest_degree - 2 <= degree or not vertices_with_degree[lowest_degree]
            self.fill_up_lowest_degree(lowest_degree, max_weight, min_weight, rank, vertices_with_degree)
            lowest_degree += 1

    def create_random_uniform_regular_graph_connect_in_the_end(self, number_vertices, rank, degree, min_weight,
                                                               max_weight):  # degree >= 2, random weights, uniform distribution. TODO: other distributions
        self.number_vertices = number_vertices
        # create vertices
        for _ in range(number_vertices):
            new_vertex = Vertex(self)
            self.vertices.add(new_vertex)
        #     self.connection_components.add(new_vertex.connection_component)

        if not (
                1. * self.number_vertices * degree / rank).is_integer():  # number_vertices * degree = number_edges * rank
            raise ValueError('no %s -uniform %s -regular connected hypergraph with &s vertices exists', rank, degree,
                             self.number_vertices)
        if not (degree >= 2 and rank >= 2):
            raise ValueError('rank and degree have to be at least 2')

        assert len(self.vertices) == self.number_vertices and not self.edges

        vertices_with_degree = [set() for i in range(degree + 1)]  # for ranks  0, 1, ..., rank

        vertices_with_degree[0] = self.vertices.copy()

        # create first edge of spanning tree
        first_edge_vertices = set(random.sample(vertices_with_degree[0], rank))
        first_edge = Edge(self, random.uniform(min_weight, max_weight), first_edge_vertices)
        self.edges.add(first_edge)

        vertices_with_degree[0].difference_update((first_edge_vertices))
        vertices_with_degree[1].update(first_edge_vertices)

        # create spanning tree
        while len(vertices_with_degree[0]) >= rank - 1:
            connected_vertex = random.sample(vertices_with_degree[1], 1)[
                0]  # more efficient than random.choice (needs set to be converted to list)
            nonconnected_vertices = set(random.sample(vertices_with_degree[0], rank - 1))

            next_edge_vertices = nonconnected_vertices.copy()
            next_edge_vertices.add(connected_vertex)
            next_edge = Edge(self, random.uniform(min_weight, max_weight), next_edge_vertices)
            self.edges.add(next_edge)

            vertices_with_degree[0].difference_update((nonconnected_vertices))
            vertices_with_degree[1].update(nonconnected_vertices)

            vertices_with_degree[1].remove(connected_vertex)
            vertices_with_degree[2].add(connected_vertex)

        # in case some vertices are 'left' over, add an edge between them and those which are already connected
        self.fill_up_lowest_degree(0, max_weight, min_weight, rank, vertices_with_degree)

        lowest_degree = 1
        while lowest_degree < degree:
            while len(vertices_with_degree[lowest_degree]) >= rank:
                next_edge_vertices = set(random.sample(vertices_with_degree[lowest_degree], rank))
                next_edge = Edge(self, random.uniform(min_weight, max_weight), next_edge_vertices)
                self.edges.add(next_edge)
                vertices_with_degree[lowest_degree].difference_update((next_edge_vertices))
                vertices_with_degree[lowest_degree + 1].update(next_edge_vertices)
            assert lowest_degree - 2 <= degree or not vertices_with_degree[lowest_degree]
            self.fill_up_lowest_degree(lowest_degree, max_weight, min_weight, rank, vertices_with_degree)
            lowest_degree += 1

    def fill_up_lowest_degree(self, lowest_degree, max_weight, min_weight, rank, vertices_with_degree):
        if vertices_with_degree[lowest_degree]:
            number_vertices_left = len(vertices_with_degree[lowest_degree])
            assert number_vertices_left >= 1 and number_vertices_left < rank

            vertices_with_current_degree = vertices_with_degree[lowest_degree].copy()
            vertices_with_higher_degree = set(random.sample(vertices_with_degree[lowest_degree + 1],
                                                            rank - number_vertices_left))

            next_edge_vertices = vertices_with_current_degree.copy()
            next_edge_vertices.update(vertices_with_higher_degree)

            next_edge = Edge(self, random.uniform(min_weight, max_weight), next_edge_vertices)
            self.edges.add(next_edge)

            vertices_with_degree[lowest_degree + 1].update(vertices_with_current_degree)
            vertices_with_degree[lowest_degree] = []

            vertices_with_degree[lowest_degree + 1].difference_update(vertices_with_higher_degree)
            vertices_with_degree[lowest_degree + 2].update(vertices_with_higher_degree)

    # def get_vertices_nonfull_degree(self, degree):
    #     nonfull_degree_vertices = [v for v in self.vertices if v.degree < degree]  # expensive
    #
    # def create_random_uniform_regular_connected_graph_without_rank_ordering(self, number_vertices, rank, degree,
    #                                                                         min_weight,
    #                                                                         max_weight):  # degree >= 2, random weights, uniform distribution. TODO: other distributions
    #     if not (1. * number_vertices * degree / rank).is_integer():  # number_vertices * degree = number_edges * rank
    #         raise ValueError('no %s -uniform %s -regular connected hypergraph with &s vertices exists', rank, degree,
    #                          number_vertices)
    #     if not (degree > 2 and rank > 2):
    #         raise ValueError('rank and degree have to be at least 2')
    #
    #     assert not self.vertices and not self.edges
    #
    #     created_vertices = 0
    #     nonfull_degree_vertices = []  # keep track of vertices, which still accept new edges
    #
    #     # create first edge of spanning tree
    #     first_edge_vertices = []
    #     for _ in range(rank):
    #         new_vertex = Vertex(self)
    #         created_vertices += 1
    #         self.vertices.append(new_vertex)
    #         first_edge_vertices.append(new_vertex)
    #         nonfull_degree_vertices.append(new_vertex)
    #     first_edge = Edge(self, random.uniform(min_weight, max_weight), first_edge_vertices)
    #     self.edges.append(first_edge)
    #     # all vertices in nonfull_degree_vertices have degree 1 here, so no need to update
    #
    #     # add vertices with edges to spanning tree
    #     while created_vertices <= number_vertices - (rank - 1):
    #
    #         # chose one vertex in the tree to which the new vertices will be appended with an edge
    #         tree_vertex = random.choice(nonfull_degree_vertices)
    #
    #         next_edge_vertices = [tree_vertex]
    #         for _ in range(rank - 1):
    #             new_vertex = Vertex(self)
    #             created_vertices += 1
    #             self.vertices.append(new_vertex)
    #             next_edge_vertices.append(new_vertex)
    #             nonfull_degree_vertices.append(new_vertex)
    #
    #         next_edge = Edge(self, random.uniform(min_weight, max_weight), next_edge_vertices)
    #         self.edges.append(next_edge)
    #         if tree_vertex.degree == degree:  # update nonfull_degree_vertices for the tree_vertex. The others have degree = 1
    #             nonfull_degree_vertices.remove(tree_vertex)
    #
    #     if created_vertices != number_vertices:  # create last vertices and append them to the graph with more than one vertex from the tree
    #         to_be_created_vertices = number_vertices - created_vertices
    #         assert (to_be_created_vertices > 0)
    #         assert to_be_created_vertices < rank - 1
    #
    #         last_edge_tree_vertices = random.sample(nonfull_degree_vertices, rank - to_be_created_vertices)
    #         last_edge_vertices = last_edge_tree_vertices.copy()
    #         for _ in range(to_be_created_vertices):
    #             new_vertex = Vertex(self)
    #             created_vertices += 1
    #             self.vertices.append(new_vertex)
    #             last_edge_vertices.append(new_vertex)
    #             nonfull_degree_vertices.append(new_vertex)
    #
    #         last_edge = Edge(self, random.uniform(min_weight, max_weight), last_edge_vertices)
    #         self.edges.append(last_edge)
    #
    #         # update nonfull_degree_vertices for the last_edge_tree_vertices. The others have degree = 1
    #         for vertex in last_edge_tree_vertices:  # todo: efficient? O(rank) < O(number_vertices)
    #             if vertex.degree == degree:
    #                 nonfull_degree_vertices.remove(
    #                     vertex)  # this line is executed O(rank) times, and each time it causes O(number_vertices) operations.
    #                 #  So total O(rank*number_vertices)
    #
    #     while nonfull_degree_vertices:  # while there are vertices with rank < n,
    #         assert len(nonfull_degree_vertices) >= rank
    #
    #         next_edge_vertices = random.sample(nonfull_degree_vertices, rank)
    #         new_edge = Edge(self, random.uniform(min_weight, max_weight), next_edge_vertices)
    #         self.edges.append(new_edge)
    #
    #         # update nonfull_degree_vertices for the next_edge_vertices
    #         for vertex in next_edge_vertices:  # todo: efficient? O(rank) < O(number_vertices)
    #             if vertex.degree == degree:
    #                 nonfull_degree_vertices.remove(
    #                     vertex)  # this line is executed O(rank) times, and each time it causes O(number_vertices) operations.
    #                 #  So total O(rank*number_vertices)

    def weight_vertices_subset(self, vertices):
        weight = 0
        for vertex in vertices:
            weight += vertex.weight
        return weight

    def intersecting_edges(self, vertices):
        intersecting_edges = set()
        for edge in self.edges:
            intersection_length = len(edge.vertices.intersection(vertices))
            if 1 <= intersection_length < len(edge.vertices):
                intersecting_edges.add(edge)
        return intersecting_edges

    def weight_edges(self, edges):
        weight = 0
        for edge in edges:
            weight += edge.weight
        return weight

    def edge_expansion(self, vertices):
        assert len(vertices) != 0
        intersecting_edges = self.intersecting_edges(vertices)
        weight_intersecting_edges = self.weight_edges(intersecting_edges)
        weight_vertices = self.weight_vertices_subset(vertices)
        return weight_intersecting_edges / weight_vertices

    def bigger_edge_expansion(self, vertices):
        intersecting_edges = self.intersecting_edges(vertices)
        weight_intersecting_edges = self.weight_edges(intersecting_edges)
        weight_vertices = self.weight_vertices_subset(vertices)
        weight_remaining_vertices = self.total_vertex_weight - weight_vertices

        return max(weight_intersecting_edges / weight_vertices, weight_intersecting_edges / weight_remaining_vertices)

    def smaller_edge_expansion(self, vertices):
        intersecting_edges = self.intersecting_edges(vertices)
        weight_intersecting_edges = self.weight_edges(intersecting_edges)
        weight_vertices = self.weight_vertices_subset(vertices)
        weight_remaining_vertices = self.total_vertex_weight - weight_vertices

        return min(weight_intersecting_edges / weight_vertices, weight_intersecting_edges / weight_remaining_vertices)

    def brute_force_hypergraph_expansion(self):
        lowest_expansion = 1
        best_set = None
        for length in range(1,
                            self.number_vertices - 1):  # range(1, int(math.ceil(len(self.vertices) / 2))): todo: why not?
            for subset_candidate in itertools.combinations(self.vertices, length):
                expansion = self.bigger_edge_expansion(subset_candidate)
                if expansion < lowest_expansion:
                    lowest_expansion = expansion
                    best_set = subset_candidate
                    print("new lowest expansion: " + (str(lowest_expansion)))

        print("lowest expansion (max of both sides) found: " + str(lowest_expansion) + " for " + str(best_set))

    def brute_force_smallest_hypergraph_expansion(self):
        lowest_expansion = 1
        best_set = None
        for length in range(1,
                            self.number_vertices - 1):  # range(1, int(math.ceil(len(self.vertices) / 2))): todo: why not?
            for subset_candidate in itertools.combinations(self.vertices, length):
                expansion = self.smaller_edge_expansion(subset_candidate)
                if expansion < lowest_expansion:
                    lowest_expansion = expansion
                    best_set = subset_candidate
                    print("new lowest expansion: " + (str(lowest_expansion)))

        print("lowest expansion (min of both sides) found: " + str(lowest_expansion) + " for " + str(best_set))

    def discrepancy_ratio(self, vertex_vector):
        nominator = 0.
        for edge in self.edges:
            max_edge_discrepancy = 0.
            for vertex_combination in itertools.combinations(edge.vertices, 2):
                vertex_combination = list(vertex_combination)

                discrepancy = math.fabs(
                    vertex_vector.vector[vertex_combination[0]] - vertex_vector.vector[vertex_combination[1]]) ** 2
                if discrepancy > max_edge_discrepancy:
                    max_edge_discrepancy = discrepancy

            nominator += edge.weight * max_edge_discrepancy

        denominator = 0.
        for vertex in self.vertices:
            denominator += vertex.weight * vertex_vector.vector[vertex] ** 2

        return nominator / denominator

    def generate_small_discrepancy_ratio_vertex_vectors(self, k):  # algorithm 3

        number_vertices = len(self.vertices)
        W = np.zeros(shape=(number_vertices, number_vertices))
        for i, vertex in enumerate(self.vertices):  # todo assert vertices are a list
            W[i, i] = vertex.weight

        # f = [np.ones((number_vertices)) for i in range(k)]
        f = []

        f.append(np.ones(number_vertices))
        f[0] /= self.weighted_norm(W, f[0])  # f_1 = 1_vector / ||1_vector||_w

        number_constructed_vectors = 1
        # g = {}  # SDP 8.3
        while number_constructed_vectors < k:
            # f[number_constructed_vectors] = self.generate_and_solve_sdp(W, vertices_list, number_vertices)
            # g = np.random.rand(number_vertices * number_vertices) #todo: evaluate whether better
            g = np.ones(number_vertices * number_vertices)

            constr = [{'type': 'eq', 'fun': lambda x: self.weighted_vertex_g_sum(x) - 1}]
            # for i in range(number_constructed_vectors):  # todo: index off by 1?
            # for j in range(number_vertices):  # todo: explicitly note down  equality 8.2
            #     # j = 0
            #     constr.append({'type': 'eq', 'fun': lambda x: self.weighted_vertex_g_f_sum(x, f[i], j)})
            # constr.append({'type': 'eq', 'fun': lambda x: self.weighted_vertex_g_f_sum_alternative(x, f[i])})
            constr.append(
                {'type': 'eq', 'fun': lambda x: self.weighted_vertex_g_f_sum_alternative_f_as_list_of_np_arrays(x, f)})
            g = scipy.optimize.minimize(fun=self.sdp_val_function, x0=g, constraints=constr,
                                        # method='SLSQP', # not working: 'nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'dogleg',    'l-bfgs-b', 'tnc'  , 'trust-ncg', cobyla
                                        options={"maxiter": 200, "disp": True})
            assert g.success
            g = np.reshape(g.x, (number_vertices, number_vertices))  # todo: transpose?

            z = np.random.normal(loc=0, scale=1, size=(number_vertices))
            new_f = np.zeros(number_vertices)
            for i, vertex in enumerate(
                    self.vertices):  # todo: check whether g needs transposing and whether the vertex-order is correct
                np.put(new_f, i, np.multiply(g[i], z))

            f.append(new_f)
            number_constructed_vectors += 1

        # todo: test whether the vectors are (at least mostly?) orthogonal : assert self.weighted_inner_product(W , f[0], f[1]) < small_threshold (0.02)

        for g, h in itertools.combinations(f, 2):
            assert self.weighted_inner_product(W, g, h) < 0.0001  # todo: adjust threshold
        return f

    def weighted_vertex_g_sum(self, g):
        value = 0
        for vertex in self.vertices:
            g_norm = np.linalg.norm(g[vertex.index * len(self.vertices):(vertex.index + 1) * len(self.vertices)]) ** 2
            value += vertex.weight * g_norm

        return value

    def weighted_vertex_g_f_sum(self, g, f, j):
        value = 0  # np.zeros(self.number_vertices)
        for vertex in self.vertices:
            value += vertex.weight * f[vertex.index] * g[len(self.vertices) * vertex.index + j]  # todo: value[?]?

        return value

    def weighted_vertex_g_f_sum_alternative(self, g, f):
        vector = np.zeros(self.number_vertices)
        for vertex in self.vertices:
            vector += vertex.weight * f[vertex.index] * g[len(self.vertices) * vertex.index: len(self.vertices) * (
                    vertex.index + 1)]  # todo: value[?]?

        value = np.linalg.norm(vector)
        return value  # , ord= math.inf)

    def weighted_vertex_g_f_sum_alternative_f_as_list_of_np_arrays(self, g, f_list):
        total_value = 0
        for a in f_list:
            vector = np.zeros(self.number_vertices)
            for vertex in self.vertices:
                vector += vertex.weight * a[vertex.index] * g[len(self.vertices) * vertex.index: len(self.vertices) * (
                        vertex.index + 1)]  # todo: value[?]?

            value = np.linalg.norm(vector)

            total_value += value
        return total_value  # , ord= math.inf)  #todo: norm with order = 1?, norm later?

    def sdp_val_function(self, g):
        value = 0
        for edge in self.edges:
            max_g_discrepancy_in_g_values = 0
            for u, v in itertools.combinations(edge.vertices, 2):

                discrepancy = np.linalg.norm(
                    g[u.index: u.index + self.number_vertices] - g[v.index: v.index + self.number_vertices]) ** 2
                if discrepancy > max_g_discrepancy_in_g_values:
                    max_g_discrepancy_in_g_values = discrepancy

            value += edge.weight * max_g_discrepancy_in_g_values
        return value

    def generate_small_expansion_set(self, k):  # algorithm 1

        vertex_list = list(self.vertices)  # use vertex indexing? todo: make sure always same order
        self.vertices = vertex_list  # todo: get rid of this technical dept (make it a list from the beginning on)
        for i, vertex in enumerate(vertex_list):
            vertex.index = i  # todo: technical dept

        f_list = self.generate_small_discrepancy_ratio_vertex_vectors(k)  # vectors
        f_vertex_vectors_list = []
        for i in range(k):
            f_vertex_vectors_list.append(Vertex_Vector(self, f_list[i], self.vertices  # todo: technical dept
                                                       ))  # todo ensure list has same order, vertex indexing

        max_discrepancy_ratio = 0

        for f_vertex_vector in f_vertex_vectors_list:
            f_discrepancy_ratio = self.discrepancy_ratio(f_vertex_vector)  # todo: adjust vertex_vector
            if f_discrepancy_ratio > max_discrepancy_ratio:
                max_discrepancy_ratio = f_discrepancy_ratio

        # xi =  max_discrepancy_ratio

        # 1.
        u = {}
        for vertex in self.vertices:
            u[vertex] = np.zeros((k))
            for i in range(k):
                u[vertex][i] = f_vertex_vectors_list[i].vector[vertex]  # todo: more efficient

        # 2
        normalized_u = {}
        for vertex in self.vertices:
            normalized_u[vertex] = u[vertex] / np.linalg.norm(u[vertex])  # todo: check whether norm applied correctly

        # 3 random projection
        beta = 0.99
        tau = k
        orthogonal_seperator = self.create_random_orthogonal_seperator(normalized_u, beta,
                                                                       tau)  # set of (vertex, u_vertex)
        random_projection = []
        for vertex in self.vertices:
            if vertex in orthogonal_seperator:
                random_projection.append((np.linalg.norm(u[vertex]), vertex))
            else:
                random_projection.append((0, vertex))

        # sweep cut
        random_projection.sort(key=lambda tuple: tuple[0])

        sorted_vertices = [tuple[1] for tuple in random_projection if tuple[0] > 0]  # todo: maybe just remove 1?

        lowest_expansion = math.inf
        lowest_expansion_vertices = set()
        for i in range(1, min(len(random_projection),
                              self.number_vertices - 1)):  # todo: make sure not all vertices can be included (expansion 0); check for index off by 1

            vertices = set(sorted_vertices[:i])
            expansion = self.edge_expansion(vertices)
            if lowest_expansion > expansion:
                lowest_expansion = expansion
                lowest_expansion_vertices = vertices

        print("lowest expansion: " + str(lowest_expansion))
        return lowest_expansion_vertices

    #    def generate_and_solve_sdp(self, W, vertices_list, number_vertices):

    def create_random_orthogonal_seperator(self, normalized_u, beta,
                                           tau):  # fact 6.7; Lemma 18 + algortihm on p.347 of A. Louis and Y. Makarychev in "Approximation Algorithms for Hypergraph Small Set Expansion and Small Set Vertex Expansion"

        # 1
        # if tau == 4: #todo: take care of special case
        # int(math.ceil(math.log2(100) / (1 - math.log2(1 + 2 / math.log2(100))))) = 11

        if tau < 8:
            word_length = 5  # as wordlength \in log2(tau) +O(1) and wolframalpha: min{log(2, x/(1 - log(2, 1 + 2/log(2, x))))}≈4.92551 at x≈8.24288
        else:
            word_length = int(math.ceil(math.log2(tau) / (1 - math.log2(1 + 2 / math.log2(tau)))))

        assert word_length >= 1
        # 3
        vertices_words = {vertex: [] for vertex in self.vertices}
        for _ in range(word_length):
            self.assign_to_vertices(beta, normalized_u, vertices_words)  # 2

        # 4
        i = 0
        orthogonal_separator = set()

        while len(orthogonal_separator) == 0:
            if self.number_vertices >= word_length:
                picked_word = [random.randint(0, 1) for _ in range(word_length)]
            else:
                word_list = [vertices_words[vertex] for vertex in self.vertices]

                word_length.sort()
                word_list = [word for word, _ in itertools.groupby(word_list)]  # removing duplicates

                number_random_words_to_append = self.number_vertices - len(word_list)  # is >0 if duplicates existed
                for _ in range(number_random_words_to_append):
                    word_list.append([random.randint(0, 1) for _ in range(word_length)])

                picked_word = random.choice(word_list)

            # 5
            r = random.uniform(0, 1)  # todo: inaccurate implementation: 0 shouldn't be included

            # 6

            remaning_vertices = set()
            for vertex in self.vertices:
                if vertices_words[vertex] == picked_word:
                    remaning_vertices.add(vertex)

            for vertex in remaning_vertices:
                if np.linalg.norm(
                        normalized_u[vertex]) > r:  # todo: isn't that always true, as the norm is always 1 and r is <1?
                    orthogonal_separator.add(vertex)

            i += 1
        print("resampled times: " + str(i))
        return orthogonal_separator

    def assign_to_vertices(self, beta, normalized_u, vertices_words):  # lemma 18

        min_time, max_time = 0, 0
        lambda_poisson = 1 / math.sqrt(beta)

        g = np.random.normal(0, 1, len(normalized_u[self.vertices[
            0]]))  # normalized_u as a parameter? #todo: why does normalized_u.size() not work?

        #

        for vertex in self.vertices:
            time_for_vertex = np.dot(g, normalized_u[vertex])  # todo: should be a scalar
            if time_for_vertex > max_time:  # todo: beautify with max/min
                max_time = time_for_vertex
            if time_for_vertex < min_time:
                min_time = time_for_vertex

        poisson_process = Poisson_Process(min_time, max_time, lambda_poisson)

        for vertex in self.vertices:
            time_for_vertex = np.dot(g, normalized_u[vertex])
            count_for_vertex = poisson_process.get_number_events_happened_until_t(
                time_for_vertex)  # np.random.poisson(lambda_poisson, time_for_vertex)  # todo: define poisson properly, just once

            #            assert count_for_vertex.is_integer()  # todo: why warning?

            if count_for_vertex % 2 == 0:
                vertices_words[vertex].append(1)
            else:
                vertices_words[vertex].append(0)

    def weighted_norm(self, W, f):
        return math.sqrt(self.weighted_inner_product(W, f, f))

    def weighted_inner_product(self, W, f, g):
        return np.sum(np.multiply(np.multiply(np.transpose(f), W), g))  # todo: sum correct?


random.seed(123)
np.random.seed(123)

# times = []
#
# m = 15
# for i in range(6, m):
#     start_time = time.time()
#
#     graph = Graph(i)
#     graph.create_random_uniform_regular_connected_graph(3, 6, 0.1, 1.1)
#
#     graph.generate_small_expansion_set(2)
#
#     end_time = time.time()
#
#     total_time = end_time-start_time
#     times.append(total_time)
#
#     print("it took " + str(total_time) + " s generate a small expansion set for a graph of "+str(i)+ " vertices")
#
# for i in range( m-1):
#     print(" for "+str(6+i)+" vertixes it took "+ str(times[i]) + " seconds")


graph = Graph()
# graph.create_random_uniform_regular_connected_graph(7, 3, 6, 0.1, 1.1)
# graph.create_random_graph_by_randomly_adding_edges(7, 3, 5, 0.1, 1.1)
graph.create_random_uniform_regular_graph_until_connected(10, 3, 6, 0.1, 1.1)

graph.generate_small_expansion_set(2)  # todo: make work for 4

graph.brute_force_smallest_hypergraph_expansion()
graph.brute_force_hypergraph_expansion()

random_vertex_vector = Vertex_Vector(graph)
discrepancy_ratio = graph.discrepancy_ratio(random_vertex_vector)
print("Discrepancy ratio ", discrepancy_ratio)


creation_algorithms =[(Graph.create_random_uniform_regular_graph_until_connected, 10, 3, 6, 0.1, 1.1), (Graph.create_random_uniform_regular_connected_graph,7, 3, 6, 0.1, 1.1)] #todo: extend
graphs_per_algorithm = 5
random_small_expansion_tries_per_graph = 5

for creation_algorithm, params in creation_algorithms: #todo: log everything, store
    for _ in range(graphs_per_algorithm):
        graph = Graph()
        graph.creation_algorithm(params)

        graph.brute_force_smallest_hypergraph_expansion()
        graph.brute_force_smallest_hypergraph_expansion()
        graph.brute_force_hypergraph_expansion_on_vertices(len(small_expansion))
        for _ in range(random_small_expansion_tries_per_graph):
            small_expansion = graph.generate_small_expansion_set()


#plot


#evaluation:
# todo: brute force time eval on 10 random graphs (degrees?)
# todo: for different algorithms to create graphs on 25 vertices
# evaluate best and 

# generate x different graphs with each generation method

# brute force all the graphs

# use approcimation algorithm (... times for each graph, take best result

# plot result


#
# repetitions = 5
# construct_time_total = 0
# brute_force_time_total = 0
#
# for _ in range(repetitions):
#     start_time = time.time()
#     graph = Graph(10)
#     graph.create_random_uniform_regular_connected_graph(5, 10, 0.1, 1.1)
#     end_time = time.time()
#     construct_time = end_time - start_time
#     construct_time_total += construct_time
#     print("it took " + str(construct_time) + " s to create " + str(graph))
#
#     graph.brute_force_hypergraph_expansion()
#     brute_force_time = time.time() - end_time
#     print("it took " + str(brute_force_time) + " s to brute-force " + str(graph))
#     brute_force_time_total += brute_force_time
#
# print("avg construct time: " + str(construct_time_total / repetitions) + ", avg brute-force time: " + str(
#     brute_force_time_total / repetitions))

# # create vertices
# vertices_to_be_connected = []
# for _ in range(number_vertices):
#     new_vertex = Vertex(self)
#     vertices_to_be_connected.append(new_vertex)
#
# # create first edge of spanning tree
# first_edge_vertices = random.choice(vertices_to_be_connected, rank)
# first_edge = Edge(self, random.uniform(min_weight, max_weight), first_edge_vertices)
# self.edges.append(first_edge)
#
# # add other vertices to spanning tree
# while

#
# def create_random_uniform_connected_graph(self, edge_size, number_vertices):
#     non_connected_vertices=[]
#     for _ in range(number_vertices):
#         vertex = Vertex(self)
#         non_connected_vertices.append(vertex)
#
#
#     #todo: add first edge of component here
#
#
#     while non_connected_vertices: #while not empty
#         current_vertex = non_connected_vertices.pop()
#
#         edge_vertices = random.sample(self.vertices, edge_size-1)
#         self.vertices.append(vertex)
#
#
#
#
# def add_random_vertex(self, edge_weight, edge_size):
#
#     # adding vertex, same connection component
#
#     new_vertex = Vertex(graph=self)
#     if len(self.vertices) > 1:  # if possible, add edge
#         if len(self.vertices) >= edge_size - 1:
#             number_old_vertices_in_new_edge = edge_size - 1
#         else:  # add as many vertices as possible to the edge
#             number_old_vertices_in_new_edge = len(self.vertices)
#
#         edge_vertices = random.sample(self.vertices, number_old_vertices_in_new_edge)
#         edge_vertices.append(new_vertex)
#         new_edge = Edge(self, edge_weight, edge_vertices)
#         self.edges.append(new_edge)
#     self.vertices.append(new_vertex)
#
# def extend_randomly(self, num_vertices, min_edge_size):
#
#
# def add_random_edge(self, edge_size):
#     if edge_size > self.max_edge_size:
#         raise ValueError("edge size too large")
#     if edge_size < len(self.vertices):
#         raise ValueError("there are not enough vertices for an edge of this size")
