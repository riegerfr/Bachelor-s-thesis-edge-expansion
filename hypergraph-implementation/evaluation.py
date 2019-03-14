import pickle
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import scipy.stats as stats
from graph import Graph
import math
import scipy.optimize

from log import Log

min_weight = 0.1
max_weight = 1.1


def evaluate_creation_algorithm(log_path=None, log_filename="creation_algorithm_log", number_vertices=12, rank=3,
                                degree=3,
                                graphs_per_algorithm=5,  # todo: 10?
                                plot=False):
    creation_algorithms_names = get_creation_algorithms(number_vertices, rank, degree)

    log = {"log_list": []}
    start_time = time.time()
    for creation_algorithm, creation_algorithm_name in creation_algorithms_names:  # todo: log everything, store

        algorithm_start_time = time.time()

        print("creation algorithm " + str(creation_algorithm_name))
        for j in range(graphs_per_algorithm):
            graph_start_time = time.time()

            print("graph " + str(j))
            graph = Graph()
            creation_algorithm(graph)

            creation_time = time.time() - graph_start_time
            print("For this graph, creation took " + str(creation_time) + " seconds ")

            brute_force_start_time = time.time()
            brute_force_best_sets, brute_force_lowest_expansion_values, brute_force_average_expansion_values, brute_force_median_expansion_values, brute_force_smallest_percentile_expansion \
                = graph.brute_force_hypergraph_expansion_each_size(use_one_sided_evaluator=False)
            # graph.brute_force_hypergraph_expansion_on_vertices(len(small_expansion))
            brute_force_time = time.time() - brute_force_start_time
            print("For this graph, brute-forcing took " + str(brute_force_time) + " seconds ")

            graph_total_time = time.time() - graph_start_time
            print("For this graph, it took " + str(graph_total_time) + " seconds in total")
            log["log_list"].append(Log(creation_algorithm_number=creation_algorithm_name, graph_number=j,
                                       graph=graph, brute_force_best_sets=brute_force_best_sets,
                                       brute_force_lowest_expansion_values=brute_force_lowest_expansion_values,
                                       brute_force_average_expansion_values=brute_force_average_expansion_values,
                                       brute_force_median_expansion_values=brute_force_median_expansion_values,
                                       creation_time=creation_time, brute_force_time=brute_force_time,
                                       graph_total_time=graph_total_time,
                                       brute_force_smallest_percentile_expansion=brute_force_smallest_percentile_expansion))
            save_log(log, log_filename, log_path)
            if plot:
                plot_creation_algorithm_differences(log_path, log_filename)
        algorithm_time = time.time() - algorithm_start_time
        print("For this creation algorithm, it took " + str(algorithm_time) + " seconds")
        log["time_algorithm " + str(creation_algorithm_name)] = algorithm_time
        save_log(log, log_filename, log_path)
    total_time = time.time() - start_time
    print("In total, it took " + str(total_time) + " seconds")
    log["total_time"] = total_time
    save_log(log, log_filename, log_path)

    return log


def evaluate_quality(log_path=None, log_filename="quality_evaluation_log_called", number_vertices=20, rank=3, degree=3,
                     graphs_per_algorithm=5,  # todo: 10?
                     random_small_expansion_tries_per_graph=100, k=2, plot=False, random_repetitions_gaussian=100):
    creation_algorithms_names = get_creation_algorithms(number_vertices, rank, degree)

    log = {"log_list": []}
    start_time = time.time()
    for creation_algorithm, creation_algorithm_name in creation_algorithms_names:  # todo: log everything, store

        algorithm_start_time = time.time()

        print("creation algorithm " + str(creation_algorithm_name))
        for j in range(graphs_per_algorithm):
            graph_start_time = time.time()

            print("graph " + str(j))
            graph = Graph()
            creation_algorithm(graph)

            creation_time = time.time() - graph_start_time
            print("For this graph, creation took " + str(creation_time) + " seconds ")

            brute_force_start_time = time.time()
            brute_force_best_sets, brute_force_lowest_expansion_values, brute_force_average_expansion_values, brute_force_median_expansion_values, brute_force_smallest_percent_expansion \
                = graph.brute_force_hypergraph_expansion_each_size()
            # graph.brute_force_hypergraph_expansion_on_vertices(len(small_expansion))
            brute_force_time = time.time() - brute_force_start_time
            print("For this graph, brute-forcing took " + str(brute_force_time) + " seconds ")

            small_expansion_vertices_list = []
            small_expansion_value_list = []
            c_estimates = []
            small_expansion_times = []
            small_expansion_start_time = time.time()

            result = graph.generate_small_expansion_set(
                k, random_small_expansion_tries_per_graph,
                random_repetitions_gaussian=random_repetitions_gaussian)  # todo: use different k_s
            for small_expansion_vertices, small_expansion_value, c_estimate in result:
                small_expansion_vertices_list.append(small_expansion_vertices)
                small_expansion_value_list.append(small_expansion_value)
                c_estimates.append(c_estimate)

                print(
                    "The found small expansion is on " + str(
                        len(small_expansion_vertices)) + " vertices with value " + str(
                        small_expansion_value
                    ) + " . The smallest expansion would have been " + str(
                        brute_force_lowest_expansion_values[len(small_expansion_vertices)]))

            small_expansion_time = time.time() - small_expansion_start_time
            print("For this graph this small expansion calculation took " + str(
                small_expansion_time))
            small_expansion_times.append(small_expansion_time)
            graph_total_time = time.time() - graph_start_time
            print("For this graph, it took " + str(graph_total_time) + " seconds in total")
            log["log_list"].append(Log(creation_algorithm_name, j,
                                       graph, brute_force_best_sets, brute_force_lowest_expansion_values,
                                       brute_force_average_expansion_values, brute_force_median_expansion_values,
                                       small_expansion_vertices_list,
                                       small_expansion_value_list, creation_time, small_expansion_times,
                                       brute_force_time,
                                       graph_total_time, k, c_estimates, brute_force_smallest_percent_expansion))
            save_log(log, log_filename, log_path)
            if plot:
                plot_values(log_path, log_filename)
        algorithm_time = time.time() - algorithm_start_time
        print("For this creation algorithm, it took " + str(algorithm_time) + " seconds")
        log["time_algorithm " + creation_algorithm_name] = algorithm_time
        save_log(log, log_filename, log_path)
    total_time = time.time() - start_time
    print("In total, it took " + str(total_time) + " seconds")
    log["total_time"] = total_time
    save_log(log, log_filename, log_path)

    return log


def get_creation_algorithms(number_vertices=12, rank=3, degree=3):
    creation_algorithms = [  # todo: beautyfiy code, extract this
        (lambda graph: graph.create_connected_graph_random_edge_adding_resampling(number_vertices, rank, degree,
                                                                                  min_weight,
                                                                                  max_weight),
         "random edges non uniform"),
        (lambda graph: graph.create_connected_graph_low_degrees_shuffel_edges_until_connected(number_vertices, rank,
                                                                                              degree,
                                                                                              min_weight,
                                                                                              max_weight),
         "shuffle edges until connected"),
        (lambda graph: graph.create_connected_graph_spanning_tree_low_degrees(number_vertices, rank, degree, min_weight,
                                                                              max_weight), "spanning tree first")

    ]  # todo: extend
    return creation_algorithms


def save_log(log, log_filename, log_path):
    if log_path is not None and log_path is not None:
        filehandler = open(log_path / (log_filename + ".pkl"), 'wb')
        pickle.dump(log, filehandler)
        filehandler.close()


def analyze_run_time_number_vertices(log_path, log_filename="number_vertices_all_logs", max_number_vertices = 25, k=3):
    start_time = time.time()
    logs = {}
    for i in range(6, max_number_vertices+1):
        vertex_number_time = time.time()
        print("testing times on " + str(i) + " vertices")
        log = evaluate_quality(number_vertices=i, rank=3, degree=3,
                               graphs_per_algorithm=2,  # todo: 10?
                               random_small_expansion_tries_per_graph=20, k=k)
        logs[i] = log
        save_log(logs, log_filename, log_path)
        print("total time for  " + str(i) + " vertices: " + str(time.time() - vertex_number_time) + " s")
        print("time since start: " + str(time.time() - start_time) + " s")
        plot_times(log_path, log_filename)


def analyze_run_time_k(log_path, log_filename="k_all_logs", max_k=5, number_vertices=20):
    start_time = time.time()
    logs = {}
    for k in range(1, max_k + 1):
        vertex_number_time = time.time()
        print("testing times on " + str(number_vertices) + " vertices and k = " + str(k))
        log = evaluate_quality(number_vertices=number_vertices, rank=3, degree=3,
                               graphs_per_algorithm=2,
                               random_small_expansion_tries_per_graph=20, k=k)
        logs[k] = log
        save_log(logs, log_filename, log_path)
        print("total time for  " + str(number_vertices) + " vertices and k = " + str(k) + " : " + str(
            time.time() - vertex_number_time) + " s")
        print("time since start: " + str(time.time() - start_time) + " s")

        plot_expansion_sizes(log_path, log_filename)
        plot_times(log_path, log_filename)


def analyze_run_time_rank_degree_combinations(log_path, log_filename="rank_degree_combinations_all_logs",
                                              number_vertices=20, rank_degree_combinations=None, k = 3):
    if rank_degree_combinations == None:
        rank_degree_combinations = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (2, 4), (4, 2), (2, 8),
                                    (8, 2), (3, 6), (6, 3), (4, 8), (8, 4)]
    start_time = time.time()
    logs = {}
    for rank, degree in rank_degree_combinations:
        rank_degree_combination = time.time()
        print("testing times on for degree " + str(degree) + " and rank " + str(rank))
        log = evaluate_quality(
            number_vertices=number_vertices, rank=rank, degree=degree,
            graphs_per_algorithm=2,  # todo: 10?
            random_small_expansion_tries_per_graph=20, k=k)
        logs[(rank, degree)] = log
        save_log(logs, log_filename, log_path)
        print("total time for   " + str(degree) + " and rank " + str(rank) + " : " + str(
            time.time() - rank_degree_combination) + " s")
        print("time since start: " + str(time.time() - start_time) + " s")
        plot_times(log_path, log_filename)


def plot_times(log_path, log_filename, xlabel="", ylabel=""):
    filehandler = open(log_path / (log_filename + ".pkl"), 'rb')
    all_logs = pickle.load(filehandler)
    filehandler.close()
    # plot times:
    # times brute force
    brute_force_times_number_vertices = {}
    # time small expansion
    small_expansion_times_number_vertices = {}
    for key in all_logs.keys():  # key: number_vertices/ rank/degree combination / k
        small_expansion_times = []
        brute_force_times = []
        for graph_log in all_logs[key]["log_list"]:
            brute_force_times.append(graph_log.brute_force_time)
            small_expansion_times.extend(graph_log.small_expansion_times)
        if not isinstance(key, int):
            key = str(key)
            key = key.replace(" ", "")
        brute_force_times_number_vertices[key] = brute_force_times
        small_expansion_times_number_vertices[key] = small_expansion_times

    brute_force_times_combined = []
    small_expansion_times_combined = []
    number_vertices_combined_brute_force = []
    number_vertices_combined_small_expansion = []
    for key in brute_force_times_number_vertices.keys():
        brute_force_times_this_number = brute_force_times_number_vertices[key]
        print("brute force times for " + str(key) + " : " + str(stats.describe(brute_force_times_this_number)))
        brute_force_times_combined.extend(brute_force_times_this_number)
        number_vertices_combined_brute_force.extend([key] * len(brute_force_times_this_number))

    for key in small_expansion_times_number_vertices.keys():
        small_expansion_times_this_number = small_expansion_times_number_vertices[key]
        print("small expansion times for " + str(key) + " : " + str(stats.describe(small_expansion_times_this_number)))
        small_expansion_times_combined.extend(small_expansion_times_this_number)
        number_vertices_combined_small_expansion.extend([key] * len(small_expansion_times_this_number))

    # def func(n, t, t0) :
    #     return np.power(n, t)+t0
    # curve_fit_small_expansion = scipy.optimize.curve_fit(func, number_vertices_combined_small_expansion, small_expansion_times_combined)
    # print("curve fit for small expansion: "+str(curve_fit_small_expansion))
    plt.scatter(number_vertices_combined_brute_force, brute_force_times_combined, label="brute-force algorithm")
    plt.scatter(number_vertices_combined_small_expansion, small_expansion_times_combined,
                label="approximation algorithm")
    plt.legend(loc='best')
    if isinstance(number_vertices_combined_brute_force[0], int):
        plt.xticks(range(math.floor(min(number_vertices_combined_brute_force)),
                         math.ceil(max(number_vertices_combined_brute_force)) + 1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(log_path / (log_filename + ".pdf"))
    plt.show()
    plt.close()


def plot_values(log_path, log_filename):
    filehandler = open(log_path / (log_filename + ".pkl"), 'rb')
    all_logs = pickle.load(filehandler)
    filehandler.close()
    # plot times:
    small_expansion_values = []
    small_expansion_vertices_numbers = []

    lowest_expansion_values_for_same_size = []
    average_expansion_values_for_same_size = []
    median_expansion_values_for_same_size = []
    smallest_percentile_expansion_values_for_same_size = []
    c_estimates = []

    graph_logs = all_logs['log_list']

    best_small_expansions_each_size = []
    respective_smallest_percentile = []

    for graph_log in graph_logs:  # key: number_vertices/ rank/degree combination / k
        small_expansion_values.extend(graph_log.small_expansion_value_list)
        c_estimates.extend(graph_log.c_estimates)

        small_expansion_vertices_numbers_this_graph = []
        for small_expansion_vertices in graph_log.small_expansion_vertices_list:
            expansion_size = len(small_expansion_vertices)
            lowest_expansion_values_for_same_size.append(
                graph_log.brute_force_lowest_expansion_values[expansion_size])
            average_expansion_values_for_same_size.append(
                graph_log.brute_force_average_expansion_values[expansion_size])
            median_expansion_values_for_same_size.append(
                graph_log.brute_force_median_expansion_values[expansion_size])
            smallest_percentile_expansion_values_for_same_size.append(
                graph_log.brute_force_smallest_percentile_expansion[expansion_size])
            small_expansion_vertices_numbers_this_graph.append(expansion_size)
        small_expansion_vertices_numbers.extend(small_expansion_vertices_numbers_this_graph)

        for expansion_size in set(
                small_expansion_vertices_numbers_this_graph):  # [len(small_expansion_vertices) for small_expansion_vertices in graph_log.small_expansion_vertices_list]:
            indices_of_size = [i for i, n in enumerate(small_expansion_vertices_numbers_this_graph) if
                               n == expansion_size]  # todo: simplify size/number
            small_expansion_values_of_size = [graph_log.small_expansion_value_list[i] for i in indices_of_size]
            best_small_expansions_each_size.append(min(small_expansion_values_of_size))
            respective_smallest_percentile.append(graph_log.brute_force_smallest_percentile_expansion[expansion_size])

    ax = plt.figure().gca()
    ax.hist(c_estimates)
    print("c estimates:"+str(stats.describe(c_estimates)))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('C values')
    plt.ylabel('Count')
    plt.savefig(log_path / (log_filename + "_C_estimates.pdf"))
    plt.show()
    plt.close()

    plt.hist(small_expansion_vertices_numbers,
             bins=np.arange(1, max(small_expansion_vertices_numbers) + 1, 1))

    plt.xlabel('number of vertices in approximate small expansion')
    plt.ylabel('count')
    plt.savefig(log_path / (log_filename + "_small_expansion_sizes.pdf"))
    plt.show()
    plt.close()

    plt.subplot(2,2,1)
    plt.plot([0, 1], [0, 1], ls="--", c=".3")
    plt.scatter(small_expansion_values, lowest_expansion_values_for_same_size,alpha = 0.5,
                label="small expansion vs lowest expansion")
    plt.xlabel('small set approximation')
    plt.ylabel('brute-force')

    plt.subplot(2,2,2)
    plt.plot([0, 1], [0, 1], ls="--", c=".3")

    plt.scatter(small_expansion_values, average_expansion_values_for_same_size,alpha = 0.5,
                label="average brute-force")

    plt.xlabel('small set approximation')
    plt.ylabel('average value brute-force')
    #plt.legend('best')

    plt.subplot(2,2,3)
    plt.plot([0, 1], [0, 1], ls="--", c=".3")
    plt.scatter(small_expansion_values, median_expansion_values_for_same_size, alpha=0.5,
                label="median brute-force")
    plt.xlabel('small set approximation')
    plt.ylabel('median value brute-force')
    # plt.scatter(small_expansion_values, smallest_percentile_expansion_values_for_same_size,
    #             label="small expansion vs 1 percentile expansion")
    #plt.xlabel('small set approximation')
    #plt.ylabel('best percentile brute-force')
    plt.subplot(2,2,4)
    plt.plot([0, 1], [0, 1], ls="--", c=".3")


    plt.scatter(best_small_expansions_each_size, respective_smallest_percentile,alpha = 0.5,
                label="lowest of small expansion vs 1 percentile expansion")
    plt.xlabel('best small set approximation')
    plt.ylabel('best percentile brute-force')
   # plt.legend(loc='best')
   # plt.xlabel('expansion value small expansion set approximation')
   # plt.ylabel('expansion value brute-force')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.savefig(log_path / (log_filename + "_expansion_values_for_same_number_verticies.pdf"))
    plt.show()
    plt.close()

    small_expansion_better_than_bf = [se <= bfe for (se, bfe) in
                                      zip(best_small_expansions_each_size, respective_smallest_percentile)]
    print("rate of small expansion equal or better than bruteforce " + str(np.mean(small_expansion_better_than_bf)))


def plot_expansion_sizes(log_path, log_filename):
    filehandler = open(log_path / (log_filename + ".pkl"), 'rb')
    all_logs = pickle.load(filehandler)
    filehandler.close()
    # plot number vertices against k:
    # time small expansion
    number_vertices_depending_on_k = {}
    for key in all_logs.keys():  # key: number_vertices/ rank/degree combination / k
        number_vertices_in_expansion = []
        for graph_log in all_logs[key]["log_list"]:
            for expansion in graph_log.small_expansion_vertices_list:
                number_vertices_in_expansion.append(len(expansion))
        if not isinstance(key, int):
            key = str(key)
        number_vertices_depending_on_k[key] = number_vertices_in_expansion

    small_expansion_k_combined = []
    number_vertices_combined_small_expansion = []

    for key in number_vertices_depending_on_k.keys():
        small_expansion_times_this_number = number_vertices_depending_on_k[key]
        small_expansion_k_combined.extend(small_expansion_times_this_number)
        number_vertices_combined_small_expansion.extend([key] * len(small_expansion_times_this_number))

    plt.scatter(number_vertices_combined_small_expansion, small_expansion_k_combined,
                label="small expansion algorithm")
    plt.legend(loc='best')

    plt.savefig(log_path / (log_filename + ".pdf"))
    plt.show()
    plt.close()


def plot_creation_algorithm_differences(log_path, log_filename):
    filehandler = open(log_path / (log_filename + ".pkl"), 'rb')
    all_logs = pickle.load(filehandler)
    filehandler.close()
    brute_force_lowest_expansion_values = {}
    brute_force_average_expansion_values = {}
    brute_force_smallest_percentile_expansion = {}
    for key in all_logs.keys():  # key: number_vertices/ rank/degree combination / k
        number_vertices_in_expansion = []
        for graph_log in all_logs["log_list"]:
            if graph_log.creation_algorithm_number not in brute_force_lowest_expansion_values.keys():
                brute_force_lowest_expansion_values[graph_log.creation_algorithm_number] = []
                brute_force_average_expansion_values[graph_log.creation_algorithm_number] = []
                brute_force_smallest_percentile_expansion[graph_log.creation_algorithm_number] = []
            brute_force_lowest_expansion_values[graph_log.creation_algorithm_number].extend(
                graph_log.brute_force_lowest_expansion_values.values())
            brute_force_average_expansion_values[graph_log.creation_algorithm_number].extend(
                graph_log.brute_force_average_expansion_values)
            brute_force_smallest_percentile_expansion[graph_log.creation_algorithm_number].extend(
                graph_log.brute_force_smallest_percentile_expansion)

        if not isinstance(key, int):
            key = str(key)

    for creation_algorithm_name in brute_force_lowest_expansion_values.keys():
        plt.hist(brute_force_lowest_expansion_values[creation_algorithm_name], alpha=0.5,
                 label=str(creation_algorithm_name),
                 bins=np.arange(0, 1, 0.05))  # todo: equal buckets
        print(stats.describe(brute_force_lowest_expansion_values[creation_algorithm_name]))

    plt.legend(loc='best')
    plt.xlabel('')

    plt.savefig(log_path / (log_filename + "_lowest_expansion.pdf"))
    plt.show()
    plt.close()

    for creation_algorithm_name in brute_force_smallest_percentile_expansion.keys():
        plt.hist(brute_force_smallest_percentile_expansion[creation_algorithm_name], alpha=0.5,
                 label="smallest percentile expansion , algorithm" + str(creation_algorithm_name))
    plt.legend(loc='best')

    plt.savefig(log_path / (log_filename + ".pdf"))
    plt.show()
    plt.close()

    for creation_algorithm_name in brute_force_average_expansion_values.keys():
        plt.hist(brute_force_average_expansion_values[creation_algorithm_name], alpha=0.5,
                 label="average expansion , algorithm" + str(creation_algorithm_name))
    plt.legend(loc='best')

    plt.savefig(log_path / (log_filename + ".pdf"))
    plt.show()
    plt.close()


random.seed(123)
np.random.seed(123)

log_path = Path("logs/")
n = 20
plot_values(log_path, log_filename="quality_evaluation_log")
plot_times(log_path, log_filename="number_vertices_all_logs", xlabel="number of vertices", ylabel="time in s")
plot_expansion_sizes(log_path, log_filename="k_all_logs")
plot_creation_algorithm_differences(log_path, log_filename="creation_algorithm_log")
plot_times(log_path, log_filename="k_all_logs", xlabel="k", ylabel="time in s")
plot_times(log_path, log_filename="rank_degree_combinations_all_logs", xlabel="(rank, degree)", ylabel="time in s")

evaluate_quality(log_path, log_filename="quality_evaluation_log", number_vertices=n, rank=3, degree=3,
                 graphs_per_algorithm=3,  # todo: 10?
                 random_small_expansion_tries_per_graph=100, k=3, plot=True)
evaluate_creation_algorithm(log_path, log_filename="creation_algorithm_log", number_vertices=n, rank=3,
                            degree=3,
                            graphs_per_algorithm=10,
                            plot=True)

# todo: compare best x% to best x% of brute-force

# todo: analyze time
analyze_run_time_rank_degree_combinations(log_path, log_filename="rank_degree_combinations_all_logs", number_vertices=n)
analyze_run_time_number_vertices(log_path, log_filename="number_vertices_all_logs", max_number_vertices=n)
analyze_run_time_k(log_path, log_filename="k_all_logs", number_vertices=n)

print("done")

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


# graph = Graph()
# # graph.create_random_uniform_regular_connected_graph(7, 3, 6, 0.1, 1.1)
# # graph.create_random_graph_by_randomly_adding_edges(7, 3, 5, 0.1, 1.1)
# graph.create_random_uniform_regular_graph_until_connected(10, 3, 6, 0.1, 1.1)
#
# graph.generate_small_expansion_set(2)  # todo: make work for 4
#
# graph.brute_force_smallest_hypergraph_expansion()
# graph.brute_force_hypergraph_expansion()
#
# random_vertex_vector = Vertex_Vector(graph)
# discrepancy_ratio = graph.discrepancy_ratio(random_vertex_vector)
# print("Discrepancy ratio ", discrepancy_ratio)


# evaluate_time(k):

# for i in range (2, k+1):


#   min_expansion_value_found = min(small_expansion_value_list)

# plot


# evaluation:
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
