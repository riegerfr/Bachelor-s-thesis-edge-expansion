class Vertex:
    indexes = 0

    def __init__(self, graph):
        self.graph = graph
        self.edges = []
        # this takes: O(n*m*r) memory
        self.weight = 0
        self.degree = 0
        self.index = Vertex.indexes
        Vertex.indexes += 1
        self.connection_component = {self}

    def add_to_edge(self, edge):
        self.degree += 1
        if (edge in self.edges):  # O(m) time!
            raise ValueError('this edge has already been added')
        self.edges.append(edge)
        self.weight += edge.weight
        self.graph.total_vertex_weight += edge.weight

    def recompute_weights_degrees(self):
        weight = 0
        degree = 0
        for edge in self.edges:
            degree += 1
            weight += edge.weight
        self.weight = weight
        self.degree = degree
