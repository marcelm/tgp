import networkx as nx

from graphread import read_cm
from graphalgo import cached_wtp, weighted_closure_cost_connected, weighted_closure_cost
from weightedgraph import CompleteGraph


def test_read_cm_95():
    with open("data/component_0095_size_3_cutoff_1.0E-10.cm") as f:
        g = read_cm(f)
    assert g.nodes() == [0, 1, 2]
    edges = g.edges(data=True)
    assert edges == [(1, 2, 2.6575775146484375)]


def test_read_cm_99():
    with open("data/component_0099_size_5_cutoff_1.0E-10.cm") as f:
        g = read_cm(f)
    assert g.nodes() == [0, 1, 2, 3, 4]
    edges = g.edges(data=True)
    assert edges == [(0, 2, 18.602060317993164), (0, 3, 8.040958404541016), (0, 4, 1.886056900024414), (1, 2, 5.9586076736450195), (1, 3, 4.7695512771606445), (2, 3, 36.10237121582031), (2, 4, 10.244125366210938), (3, 4, 14.886056900024414)]


def test_weighted_graph_connected_components():
    with open("data/component_0099_size_5_cutoff_1.0E-10.cm") as f:
        g = read_cm(f)
    assert isinstance(g, CompleteGraph)
    subgraph = g.subgraph([0, 2])


def test_complete_graph_copy():
    with open("data/component_0021_size_19_cutoff_1.0E-10.cm") as f:
        cg = read_cm(f)
    assert len(cg.nodes()) == 19
    cg_copy = cg.copy()
    assert cg.nodes() == cg_copy.nodes()
    assert cg.edges() == cg_copy.edges()
    assert weighted_closure_cost_connected(cg_copy) == 141.90280961990356


def test_weighted_closure_cost_connected():
    with open("data/component_0021_size_19_cutoff_1.0E-10.cm") as f:
        g = read_cm(f)
    assert weighted_closure_cost_connected(g) == 141.90280961990356


def test_connected_components_after_split():
    with open("data/component_0021_size_19_cutoff_1.0E-10.cm") as f:
        g = read_cm(f)
    assert len(nx.connected_components(g)) == 1

    # Setting edge weight to zero should remove the edge for the purpose of
    # computing connected components
    g.add_edge(2, 12, 0.0)
    g.add_edge(12, 17, 0.0)
    g.add_edge(1, 12, 0.0)

    assert len(nx.connected_components(g)) == 2


def test_cached_wtp():
    with open("data/component_0021_size_19_cutoff_1.0E-10.cm") as f:
        g = read_cm(f)
    assert len(nx.connected_component_subgraphs(g)) == 1
    dels, cost = cached_wtp(g, postprocessing=True)
    assert abs(cost - 22.4704899788) <= 0.0001
    assert dels == [(2, 12), (12, 17), (1, 12)]


