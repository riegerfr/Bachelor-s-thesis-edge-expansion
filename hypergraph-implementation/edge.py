class Edge:

    def __init__(self, graph, weight, vertices):
        self.graph = graph
        if weight <= 0:
            raise ValueError('Edges always have positive weights')
        self.weight = weight
        self.vertices = vertices

        for vertex in vertices:
            vertex.add_to_edge(self)
