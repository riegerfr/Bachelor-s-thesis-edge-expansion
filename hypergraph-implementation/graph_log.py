class Graph_Log:
    def __init__(self, creation_algorithm_number=None, graph_number=None, graph=None, brute_force_best_sets=None,
                 brute_force_lowest_expansion_values=None, brute_force_average_expansion_values=None,
                 brute_force_median_expansion_values=None, small_expansion_vertices_list=None,
                 small_expansion_value_list=None, creation_time=None, small_expansion_times=None, brute_force_time=None,
                 graph_total_time=None,
                 k=None, c_estimates=None, brute_force_smallest_percentile_expansion=None):
        self.brute_force_average_expansion_values = brute_force_average_expansion_values
        self.brute_force_median_expansion_values = brute_force_median_expansion_values
        self.graph_number = graph_number
        self.creation_algorithm_number = creation_algorithm_number
        self.graph = graph
        self.brute_force_best_sets = brute_force_best_sets
        self.brute_force_lowest_expansion_values = brute_force_lowest_expansion_values
        self.small_expansion_vertices_list = small_expansion_vertices_list
        self.small_expansion_value_list = small_expansion_value_list
        self.creation_time = creation_time
        self.small_expansion_times = small_expansion_times
        self.brute_force_time = brute_force_time
        self.graph_total_time = graph_total_time
        self.k = k
        self.c_estimates = c_estimates
        self.brute_force_smallest_percentile_expansion = brute_force_smallest_percentile_expansion
