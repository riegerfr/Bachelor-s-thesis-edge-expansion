class Edge:

    def __init__(self, graph, weight, vertices):
        self.graph = graph
        if weight <= 0:
            raise ValueError('Edges always have positive weights')
        self.weight = weight
        self.vertices = vertices

        for vertex in vertices:  # todo: useful?
            vertex.add_to_edge(self)

        # possible_same_connection_component = next(iter(vertices)).connection_component
        # if any(vertex.connection_component != possible_same_connection_component for vertex in vertices):
        #     new_connection_component_vertices = vertices
        # for vertex in vertices:
        #     new_connection_component_vertices.update(vertex.connection_component.vertices) #todo: Set changed size during iteration?
        # new_connection_component = ConnectionComponent(new_connection_component_vertices)