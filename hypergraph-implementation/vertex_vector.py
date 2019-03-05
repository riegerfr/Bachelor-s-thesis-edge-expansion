import random


class Vertex_Vector:
    def __init__(self, graph, values=None, vertex_list=None):  # todo: assert values have correct order
        self.graph = graph
        self.vector = {}
        if values is not None:
            assert len(values) == len(vertex_list)
            for i, vertex in enumerate(vertex_list):
                self.vector[vertex] = values[i]
        else:
            self.insert_random_values()

    def insert_random_values(self):
        for vertex in self.graph.vertices:
            self.vector[vertex] = random.uniform(-1, 1)

    def all_ones(self):
        for vertex in self.graph.vertices:
            self.vector[vertex] = 1
