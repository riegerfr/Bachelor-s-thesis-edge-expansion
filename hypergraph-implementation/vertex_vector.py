import random


class Vertex_Vector:
    def __init__(self, graph, values=None):
        self.graph = graph
        self.vector = {}
        if values is not None:
            assert len(values) == len(graph.vertices)
            for i, vertex in enumerate(graph.vertices):
                assert i == vertex.index
                self.vector[vertex] = values[i]
        else:
            self.insert_random_values()

    def get_plain_vector(self):
        plain_vector = []
        for i, vertex in enumerate(self.graph.vertices):
            assert i == vertex.index
            plain_vector.append(self.vector[vertex])
        return plain_vector

    def insert_random_values(self):
        for vertex in self.graph.vertices:
            self.vector[vertex] = random.uniform(-1, 1)

    def all_ones(self):
        for vertex in self.graph.vertices:
            self.vector[vertex] = 1
