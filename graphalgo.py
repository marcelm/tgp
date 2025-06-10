#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import random
import sys

#from heapq import heappush, heappop, heapreplace
from bisect import bisect_left, bisect_right

from networkx import Graph

import gray
import bitsgen
import graphgen
from weightedgraph import WeightedGraph, CompleteGraph

_infinity = 1e50000

# TODO
# - aus den create_graph-Funktionen Klassen machen
# - optimize exact() by working on connected components separately

def find_culprit(g):
	"""
	Returns edge which, when deleted, would result in the greatest improvement
	to the DfT.
	"""
	assert g.number_of_edges() > 0

	# go over all edges
	maximum_deviation = -sys.maxint - 1 # TODO, g.number_of_nodes() should be enough
	for u, v in g.edges_iter():
		gu = g.adj[u]
		gv = g.adj[v]
		s = set(gu)
		s.update(gv)

		#common = len(gu) + len(gv) - len(s)
		#noncommon = len(s) - common
		#deviation = noncommon - common - 2
		deviation = 3*len(s) - 2*(len(gu) + len(gv)) - 2

		if __debug__:
			set_u = set(g[u]) - set([v])
			set_v = set(g[v]) - set([u])
		assert deviation == len(set_u ^ set_v) - len(set_u & set_v)

		#print "edge", (u,v), "dev:", deviation
		if deviation > maximum_deviation:
			maximum_deviation = deviation
			culprit = (u,v)

# 	print "culprit:",culprit,maximum_deviation
	return culprit

def closure_cost(g):
	"""Returns number of edges that need to be added in order for g to become transitive
	"""
	subgraphs = nx.connected_component_subgraphs(g)

	cost = 0
	for subgraph in subgraphs:
		n = subgraph.number_of_nodes()
		e = subgraph.number_of_edges()
		cost += n*(n-1)/2 - e
	return cost

def transitive_closure(g):
	"""TODO!"""
	subgraphs = nx.connected_component_subgraphs(g)

	for subgraph in subgraphs:
		for u in subgraph.nodes():
			for v in subgraph.nodes():
				g.add_edge(u, v)
	return g

def dft(graph, func):
	subgraphs = nx.connected_component_subgraphs(graph)
	deletions = []
	cost = 0
	for g in subgraphs:
		d, c = func(g)
		deletions.extend(d)
		cost += c
	return (deletions, cost)

def even_better_dft(graph):
	return dft(graph, _even_better_dft_connected)

def _even_better_dft_connected(graph):
	"""
	Input graph *must* be connected!

	Returns a tuple of list of edges to delete and the cost to do it
	"""

	if graph.number_of_edges() in [0, 1]:
		return ([], 0)

	# a "working copy"
	g = graph.copy()
	deletions = []
	delcount = 0
	cost = closure_cost(g)

	split = False
	while True:
		culprit = find_culprit(g)
		g.delete_edge(culprit)
		delcount += 1
		components = nx.connected_components(g)
		deletions.append(culprit)
		if len(components) == 2:
			break

	# We've got a split: Fix list of deletions such that
	# only those edges contributing to the cut are included.
	nodes1 = set(components[0])
	nodes2 = set(components[1])

	newdels = []
	for u, v in deletions:
		if (u in nodes1 and v in nodes2) or (u in nodes2 and v in nodes1):
			newdels.append( (u, v) )
	delcount = len(newdels)
	if cost <= delcount:
		return ([], cost)

	deletions = newdels
	subgraphs = map(graph.subgraph, components)

	# recursively solve the problem for the two subgraphs
	list1, cost1 = _even_better_dft_connected(subgraphs[0])
	list2, cost2 = _even_better_dft_connected(subgraphs[1])

	splitcost = cost1 + cost2 + delcount
	if splitcost < cost:
		deletions.extend(list1)
		deletions.extend(list2)
		return (deletions, splitcost)
	else:
		return ([], cost)

def perhaps_better_dft(graph):
	subgraphs = nx.connected_component_subgraphs(graph)
	deletions = []
	cost = 0
	for g in subgraphs:
		d, c = _perhaps_better_dft_connected(g, closure_cost(g))
		deletions.extend(d)
		cost += c
	return (deletions, cost)

def _perhaps_better_dft_connected(graph, maxcost):
	"""
	Input graph *must* be connected!

	Returns a tuple of list of edges to delete and the cost to do it.
	maxcost is the maximum allowed cost.
	"""
	if graph.number_of_edges() in [0, 1]:
		return ([], 0)

	# a "working copy"
	g = graph.copy()
	deletions = []
	delcount = 0
	closurecost = closure_cost(g)
	if maxcost <= 0:
		return ([], closurecost)
	maxcost = min(maxcost, closurecost)

	while True:
		culprit = find_culprit(g)
		g.delete_edge(culprit)
		delcount += 1
		components = nx.connected_components(g)
		deletions.append(culprit)
		if len(components) == 2:
			break

	# We've got a split: Fix list of deletions such that
	# only those edges contributing to the cut are included.
	nodes1 = set(components[0])
	nodes2 = set(components[1])

	newdels = []
	for u, v in deletions:
		if (u in nodes1 and v in nodes2) or (u in nodes2 and v in nodes1):
			newdels.append( (u, v) )

	delcount = len(newdels)
	if delcount >= maxcost:
		return ([], closurecost)

	deletions = newdels
	subgraphs = map(graph.subgraph, components)

	list1, cost1 = _perhaps_better_dft_connected(subgraphs[0], maxcost-delcount)
	list2, cost2 = _perhaps_better_dft_connected(subgraphs[1], maxcost-delcount)

	splitcost = cost1 + cost2 + delcount
	if splitcost < closurecost:
		deletions.extend(list1)
		deletions.extend(list2)
		return (deletions, cost1 + cost2 + delcount)
	else:
		return ([], closurecost)

##########
########## here the weighted stuff begins
##########

def wdft(graph):
	"""
	Computes the weighted deviation from transitivity for the graph.

	This is a very inefficient implementation and is intended to help
	in debugging and testing, not for production use.

	Definition:
	D(G) := \sum_\{u,v,w\}\in\ct(G) \min\{|s(uv)|, |s(vw)|, |s(uw)|\}
	"""

	deviation = 0.0
	nodes = graph.nodes()
	n = len(nodes)
	for ui in xrange(n):
		for vi in xrange(ui+1, n):
			for wi in xrange(vi+1, n):
				u, v, w = nodes[ui], nodes[vi], nodes[wi]
				# count edges in current triple
				m = 0
				for edge in [ (u, v), (v, w), (u, w) ]:
					if graph.get_edge(*edge) > 0.0:
						m += 1
				if m == 2:
					# conflict triple found
					deviation += min(abs(graph.get_edge(u, v)), abs(graph.get_edge(v, w)), abs(graph.get_edge(u, w)))
	return deviation

def weighted_closure_cost_connected(graph):
	"""
	Returns what it costs to make 'graph' transitive. graph must
	be connected.

	TODO the closure cost could be kept in the graph itself
	"""
	assert nx.is_connected(graph)

	# iterate over *all* edges, add up all those with negative weight

	cost = -sum(w for u,v,w in graph.all_edges_iter(data=True) if w < 0.0)
	assert cost >= 0.0

	if type(graph) is CompleteGraph:
		# FIXME, very WeightedGraph/CompleteGraph-specific
		# there are edges that don't exist: count them
		n = graph.number_of_nodes()
		cost += -graph.missingweight * (n*(n-1)/2 - graph.number_of_all_edges())
	return cost

def weighted_closure_cost(graph):
	"""
	Returns what it costs to make 'graph' transitive.
	"""
	subgraphs = nx.connected_component_subgraphs(graph)
	cost = 0.0
	for subgraph in subgraphs:
		cost += weighted_closure_cost_connected(subgraph)
	return cost
	#return sum(map(weighted_closure_cost_connected, nx.connected_component_subgraphs(graph)))

def score_sum(uv, vw, uw):
	"""
	uv, vw are the weights of the existing edge. uw is the weight of the missing edge.
	uv > 0, vw > 0, uw <= 0
	"""
	return sum((uv, vw, uw))

def score_min(uv, vw, uw):
	"""
	uv, vw are the weights of the existing edge. uw is the weight of the missing edge.
	uv > 0, vw > 0, uw <= 0
	"""
	return min(uv, vw, -uw)

def score_min2(uv, vw, uw):
	"""
	uv, vw are the weights of the existing edge. uw is the weight of the missing edge.
	uv > 0, vw > 0, uw <= 0
	"""
	return min(uv, vw)

def score_sumabs(uv, vw, uw):
	"""
	uv, vw are the weights of the existing edge. uw is the weight of the missing edge.
	uv > 0, vw > 0, uw <= 0
	"""
	return sum((uv, vw, -uw))/3.0

def score_missing(uv, vw, uw):
	return -uw

def weighted_find_culprit(graph, scorefunc=score_min, subtractweight=True):
	"""
	wie min, bloss dass das gewicht der zu loeschenden kante subtrahiert wird.
	Returns edge which, when deleted, would result in the greatest
	transitivity improvement.
	"""
	assert graph.number_of_edges() > 0
	# iterate over all edges
	best_ti = -_infinity

	for u, v, w in graph.edges_iter():
		ti = 0.0
		set_graph_u = set(graph.neighbors_iter(u))
		set_graph_v = set(graph.neighbors_iter(v))
		# non-common neighbors of u
		for x in set_graph_u - set_graph_v - set([v]):
			weight_xv = graph.get_edge(x, v)
			assert weight_xv <= 0.0
			ti += scorefunc(w, graph.get_edge(u, x), weight_xv)

		# non-common neighbors of v
		for x in set_graph_v - set_graph_u - set([u]):
			weight_xu = graph.get_edge(x, u)
			assert weight_xu <= 0.0
			ti += scorefunc(w, graph.get_edge(v, x), weight_xu)

		# common neighbors
		common_neighbors = set_graph_v & set_graph_u
		for x in common_neighbors:
			ti -= scorefunc(graph.get_edge(u, x), graph.get_edge(v, x), -w)
		if subtractweight:
			# penalize edge removal
			ti -= w
#		print "edge", (u,v), "ti:", ti
		if ti > best_ti:
			best_ti = ti
			culprit = (u,v)
			cc = len(common_neighbors)

# 	print "culprit:", culprit, best_ti
	return (culprit, cc, best_ti)

def weighted_find_culprits(graph):
	"""
	Returns edges, sorted by their transitivity improvement.

	FIXME TODO NOTE This does not work since this procedure
	does not delete the culprit from the graph and therefore
	the edge scores computed afterwards are wrong. (Also,
	there is an infinite loop.)
	"""
	assert graph.number_of_edges() > 0
	# iterate over all edges
	maximum_timprovement = -_infinity
	culprits = []
	edges = graph.edges_iter() # TODO warum geht nicht _iter()?
	while True:
		for u, v, w in edges:
			timprovement = 0.0
			set_graph_u = set(graph.neighbors_iter(u))
			set_graph_v = set(graph.neighbors_iter(v))
			# non-common neighbors of u
			for x in set_graph_u - set_graph_v - set([v]):
				weight_xv = graph.get_edge(x, v)
				assert weight_xv <= 0.0
				timprovement += min(w, -weight_xv, graph.get_edge(u, x))

			# non-common neighbors of v
			for x in set_graph_v - set_graph_u - set([u]):
				weight_xu = graph.get_edge(x, u)
				assert weight_xu <= 0.0
				timprovement += min(w, -weight_xu, graph.get_edge(v, x))

			# common neighbors
			common_neighbors = set_graph_v & set_graph_u
			for x in common_neighbors:
				timprovement -= min(w, graph.get_edge(u, x), graph.get_edge(v, x))

			timprovement -= w
	#		print "edge", (u,v), "ti:", timprovement
			if timprovement > maximum_timprovement:
				maximum_timprovement = timprovement
				culprit = (u, v)
				max_cc = common_neighbors
		culprits.append(culprit)
		if len(max_cc) > 0:
			edges = [ (u, x, graph.get_edge(u, x)) for x in max_cc ]
			edges.extend((v, x, graph.get_edge(v, x)) for x in max_cc)
		else:
			break
	return culprits

def _uncached_wtp_connected(graph, find_culprit_func, wfactor=-1):
	"""
	Input graph must be of type WeightedGraph and connected!

	Returns a tuple of list of edges to delete and the cost to do it
	"""
	if graph.number_of_edges() in [0, 1]:
		return ([], 0)

	# a "working copy"
	graph = graph.copy()
	deletions = []
	delcost = 0.0
	cost = weighted_closure_cost_connected(graph)

	#print "cost:", cost
	split = False
	while delcost < cost:
		(u,v), cc, ti = find_culprit_func(graph)
		print "culprit:", (u,v), ti
		w = graph.get_edge(u, v) # TODO get_edge does not take a tuple?
		delcost += w

		assert w > 0.0

		if __debug__:
			before = wdft(graph)
		# "add_edge" is misleading: the edge will be replaced
		graph.add_edge(u, v, w*wfactor)
		print "remove", (u,v)
		if __debug__:
			after = wdft(graph)
# 			print "before:", before, "after:", after, "w:", w, "before-after-w:", before-after-w, "ti:", ti
# 			print "(ti: %f)" % ti
			assert before - after - w == ti
		deletions.append( (u, v, w) )
		if cc > 0:
			# cc tells us how many common neighbors u and v have.
			# If they still have a common neighbor, then the current
			# deletion can't lead to a split of the graph.
			continue
		components = nx.connected_components(graph)
		if len(components) == 2:
			split = True
			break
		else:
			assert len(components) == 1

	if not split:
		# split is more expensive than just leaving it as is
		return ([], cost)

	# detect if a singleton is split off
# 	if len(components[1]) == 1:
# 		# trying to split off a singleton,
# 		# find out if that's a smart thing to do
# 		singleton = components[1][0]
# 		assert len(deletions) == 1
# 		u, v, w = deletions[0]
# 		assert singleton == u or singleton == v
# 		if v == singleton:
# 			u, v = v, u
# 		# u is singleton


	# we've got a split: fix list of deletions
	nodes1 = set(components[0])
	nodes2 = set(components[1])

	newdels = []
	for u,v,w in deletions:
		if (u in nodes1 and v in nodes2) or (u in nodes2 and v in nodes1):
			newdels.append((u,v))
		else:
			delcost -= w
			graph.add_edge(u, v, w)

	deletions = newdels
	subgraphs = map(graph.subgraph, components)

	# recursively solve the problem for the two subgraphs
	list1, cost1 = _uncached_wtp_connected(subgraphs[0], find_culprit_func, wfactor)
	list2, cost2 = _uncached_wtp_connected(subgraphs[1], find_culprit_func, wfactor)

	splitcost = cost1 + cost2 + delcost
	if splitcost < cost:
		deletions.extend(list1)
		deletions.extend(list2)
		return (deletions, splitcost)
	else:
		return ([], cost)

def _wtp_connected_multi(graph):
	"""
	"""
	if graph.number_of_edges() in [0, 1]:
		return ([], 0)

	# a "working copy"
	graph = graph.copy()
	deletions = []
	delcost = 0.0
	cost = weighted_closure_cost_connected(graph)

	split = False
	while delcost < cost:
		culprits = weighted_find_culprits(graph)
		for edge in culprits:
			assert nx.number_connected_components(graph) == 1
			w = graph.get_edge(u, v) # TODO get_edge does not take a tuple?
			delcost += w
			assert w > 0.0
			# "add_edge" is misleading: the edge will be replaced
			graph.add_edge(u, v, 0.0)
		deletions.append( (u, v, w) )
		components = nx.connected_components(graph)
		if len(components) == 2:
			split = True
			break
		else:
			assert len(components) == 1

	if not split:
		# split is more expensive than just leaving it as is
		return ([], cost)

	# we've got a split: fix list of deletions
	# This doesn't improving results as much as one would expect, in tests
	# the costs went down on average by less than 1%.
	nodes1 = set(components[0])
	nodes2 = set(components[1])

	newdels = []
	for u,v,w in deletions:
		if (u in nodes1 and v in nodes2) or (u in nodes2 and v in nodes1):
			newdels.append((u,v))
		else:
			delcost -= w
			graph.add_edge(u, v, w)

	deletions = newdels
	subgraphs = map(graph.subgraph, components)
# 	print "new graph:", graph.all_edges()
# 	print "new deletions:", deletions, "cost:", delcost

# 	oldlen = len(deletions)
# 	deletions = [ (u,v,w) for u,v,w in deletions if (u in nodes1 and v in nodes2) or (u in nodes2 and v in nodes1) ]
# 	print "fixed deletions:", deletions
# 	if len(deletions) != oldlen:
# 		print "fixing"
# 		delcost = sum(w for u,v,w in deletions)
# 		print "fixed delcost:", delcost
# 	deletions = [ (u,v) for u,v,w in deletions ]

	# recursively solve the problem for the two subgraphs
	list1, cost1 = _weighted_dft_connected(subgraphs[0], find_culprit_func)
	list2, cost2 = _weighted_dft_connected(subgraphs[1], find_culprit_func)

	splitcost = cost1 + cost2 + delcost
	if splitcost < cost:
		deletions.extend(list1)
		deletions.extend(list2)
		return (deletions, splitcost)
	else:
		return ([], cost)


def _wtp_connected_multi(graph):
	"""
	Input graph must be of type WeightedGraph and connected!

	Returns a tuple of list of edges to delete and the cost to do it
	"""
	if graph.number_of_edges() in [0, 1]:
		return ([], 0)

	# a "working copy"
	graph = graph.copy()
	deletions = []
	delcost = 0.0
	cost = weighted_closure_cost_connected(graph)

	split = False
	#improvements = list(weighted_find_culprits(graph)) # TODO
# 	print "components:", len(nx.connected_components(graph))
# 	print "improvements:", list(improvements)

	for _, u, v in weighted_find_culprits(graph):
		if delcost >= cost:
			break
		w = graph.get_edge(u, v) # TODO get_edge does not take a tuple?
		delcost += w

		assert w > 0.0
		# "add_edge" is misleading: the edge will be replaced
		graph.add_edge(u, v, 0.0)
		deletions.append( (u, v, w) )
		components = nx.connected_components(graph)
		if len(components) == 2:
			split = True
			break
		else:
			assert len(components) == 1

	if not split:
		# split is more expensive than just leaving it as is
		return ([], cost)

	# we've got a split: fix list of deletions
	nodes1 = set(components[0])
	nodes2 = set(components[1])

	newdels = []
	for u,v,w in deletions:
		if (u in nodes1 and v in nodes2) or (u in nodes2 and v in nodes1):
			newdels.append((u,v))
		else:
			delcost -= w
			graph.add_edge(u, v, w)

	deletions = newdels
	subgraphs = map(graph.subgraph, components)

	# recursively solve the problem for the two subgraphs
	list1, cost1 = _weighted_dft_connected_multi(subgraphs[0])
	list2, cost2 = _weighted_dft_connected_multi(subgraphs[1])

	splitcost = cost1 + cost2 + delcost
	if splitcost < cost:
		deletions.extend(list1)
		deletions.extend(list2)
		return (deletions, splitcost)
	else:
		return ([], cost)

def weighted_dft_multi(graph):
	"""
	Compute solution for transitive projection using the weighted
	transitivity improvement heuristic. Input graph must be of type
	WeightedGraph. Returns a tuple (deletions, cost) where deletions is
	an edge list.
	"""
	deletions = []
	cost = 0.0
	for g in nx.connected_component_subgraphs(graph):
		d, c = _weighted_dft_connected_multi(g)
		deletions.extend(d)
		cost += c
	return (deletions, cost)

######
###### WTP with cached scores
######

class DftGraph(object):
	"""
	TODO common neighbors optimization (if #(common) > 0, then the graph surely is still connected)
	"""
	def __init__(self, graph=None):
		# for now, we store the weighted graph as a member
		if graph is None:
			self.wgraph = None
		else:
			self.wgraph = graph.copy()
		self.tigraph = Graph()

		# compute TI for all edges in the source graph and
		# store TI values for later.
		best_ti = -_infinity
		best_edge = None
		if self.wgraph is not None:
			for u, v, w in self.wgraph.edges_iter(data=True):
				ti = self.ti(u, v, w)
				self.tigraph.add_edge(u, v, ti)
				if ti > best_ti:
					best_ti = ti
					best_edge = (u, v)
		self._best_edge = best_edge
		self._best_ti = best_ti

# 	def copy(self):
# 		"""returns a copy of this graph"""
# 		graph = self.__class__()
# 		graph.wgraph = self.wgraph.copy()
# 		graph.tigraph = self.tigraph.copy()
# 		graph._best_edge = self._best_edge
# 		graph._best_ti = self._best_ti
# 		return graph

	def subgraph(self, nbunch):
		"""returns the subgraph induced by the nodes in nbunch"""
		# FIXME this works correctly only when nbunch is a connected component
		graph = self.__class__()
		graph.wgraph = self.wgraph.subgraph(nbunch)
		graph.tigraph = self.tigraph.subgraph(nbunch)
		graph._update_best()
		return graph

	def ti(self, u, v, w):
		ti = 0.0
		graph = self.wgraph
		gge = graph.get_edge
		set_graph_u = set(graph.neighbors_iter(u))
		set_graph_v = set(graph.neighbors_iter(v))
		# non-common neighbors of u
		for x in set_graph_u - set_graph_v - set([v]):
			ti += min(w, -gge(x, v), gge(u, x))

		# non-common neighbors of v
		for x in set_graph_v - set_graph_u - set([u]):
			ti += min(w, -gge(x, u), gge(v, x))

		# common neighbors
		for x in set_graph_v & set_graph_u:
			ti -= min(w, gge(u, x), gge(v, x))
		# TODO hier mal len(common_neigbors) einbauen

		return ti-w

	def get_culprit(self):
		"""
		"""
		return self._best_edge

	def get_culprit_ti(self):
		return self._best_ti

	def delete_culprit(self):
		u, v = self._best_edge

		self.tigraph.delete_edge(u, v)
		w = self.wgraph.get_edge(u, v)
		self.wgraph.add_edge(u, v, -w)

		# recompute TI values for edges to adjacent nodes

		gge = self.wgraph.get_edge
		set_graph_u = set(self.wgraph.neighbors_iter(u))
		set_graph_v = set(self.wgraph.neighbors_iter(v))

		stae = self.tigraph.add_edge
		stge = self.tigraph.get_edge

		# common neighbors
		for x in set_graph_v & set_graph_u:
			d = min(w, gge(u, x), gge(v, x))
			ti = stge(u, x)
			stae(u, x, ti+d+d)
			ti = stge(v, x)
			stae(v, x, ti+d+d)

		# non-common neighbors of u
		for x in set_graph_u - set_graph_v - set([v]):
			ti = stge(u, x)
			ti -= min(w, -gge(x, v), gge(u, x))
			stae(u, x, ti)

		# non-common neighbors of v
		for x in set_graph_v - set_graph_u - set([u]):
			ti = stge(v, x)
			ti -= min(w, -gge(x, u), gge(v, x))
			stae(v, x, ti)

		self._update_best()

	def _update_best(self):
		# find the best edge
		# TODO heap
		if self.tigraph.number_of_edges() == 0:
			self._best_edge = None
			self._best_ti = None
		else:
			best_ti, best_u, best_v = max((w, u, v) for u, v, w in self.tigraph.edges_iter(data=True))
			self._best_edge = best_u, best_v
			self._best_ti = best_ti

	def add_edge(self, u, v, w):
		self.tigraph.add_edge(u, v, self.ti(u, v, w))
		self.wgraph.add_edge(u, v, w)

		gge = self.wgraph.get_edge
		set_graph_u = set(self.wgraph.neighbors_iter(u))
		set_graph_v = set(self.wgraph.neighbors_iter(v))

		stae = self.tigraph.add_edge
		stge = self.tigraph.get_edge

		# common neighbors
		for x in set_graph_v & set_graph_u:
			d = min(w, gge(u, x), gge(v, x))
			ti = stge(u, x)
			stae(u, x, ti-d-d)
			ti = stge(v, x)
			stae(v, x, ti-d-d)

		# non-common neighbors of u
		for x in set_graph_u - set_graph_v - set([v]):
			ti = stge(u, x)
			ti += min(w, -gge(x, v), gge(u, x))
			stae(u, x, ti)

		# non-common neighbors of v
		for x in set_graph_v - set_graph_u - set([u]):
			ti = stge(v, x)
			ti += min(w, -gge(x, u), gge(v, x))
			stae(v, x, ti)

		self._update_best()

def _cached_wtp_connected(dftgraph, maxcost=None):
	"""
	Input graph must be of type WeightedGraph and connected!

	Returns a tuple of list of edges to delete and the cost to do it
	"""
	graph = dftgraph.wgraph
	if graph.number_of_edges() in [0, 1]:
		return ([], 0)

	deletions = []
	delcost = 0.0
	cost = weighted_closure_cost_connected(graph)

	if maxcost is None:
		maxcost = cost
	else:
		maxcost = min(maxcost, cost)
	if maxcost <= 0:
		return ([], cost)

	while True:
		u, v = dftgraph.get_culprit()
		if __debug__:
			ti = dftgraph.get_culprit_ti()
		w = graph.get_edge(u, v)
		delcost += w

		assert w > 0.0
		deletions.append( (u, v, w) )

		if __debug__: before = wdft(graph)
		dftgraph.delete_culprit()

		if __debug__:
			after = wdft(graph)
#			print "before:", before, "after:", after, "w:", w, "before-after-w:", before-after-w, "ti:", ti
# 			print "(ti: %f)" % ti
			assert before - after - w == ti

# 		if cc > 0:
# 			# cc tells us how many common neighbors u and v have.
# 			# If they still have a common neighbor, then the current
# 			# deletion can't lead to a split of the graph.
# 			continue
		path = nx.bidirectional_shortest_path(dftgraph.tigraph, u, v)
		if path is False:
			# split detected
			break

	components = nx.connected_components(dftgraph.tigraph) # TODO oder wgraph

	# fix list of deletions
	nodes1 = set(components[0])
	nodes2 = set(components[1])

	newdels = []
	for u,v,w in deletions:
		if (u in nodes1 and v in nodes2) or (u in nodes2 and v in nodes1):
			newdels.append( (u,v) )
		else:
			delcost -= w
			dftgraph.add_edge(u, v, w)

	if delcost >= maxcost:
		return ([], cost)

	deletions = newdels
	dftsubgraphs = map(dftgraph.subgraph, components)

	# recursively solve the problem for the two subgraphs
	list2, cost2 = _cached_wtp_connected(dftsubgraphs[1], maxcost-delcost)
	if cost2 + delcost >= cost:
		return ([], cost)
	list1, cost1 = _cached_wtp_connected(dftsubgraphs[0], maxcost-delcost)

	splitcost = cost1 + cost2 + delcost
	if splitcost < cost:
		deletions.extend(list1)
		deletions.extend(list2)
		return (deletions, splitcost)
	else:
		return ([], cost)

def postprocess(graph, deletions, cost):
	"""Filters out deletions between singletons.
	Returns corrected pair (deletions, cost).
	graph is not modified."""
	tmp = graph.copy()
	tmp.delete_edges_from(deletions)
	singletons = set(component[0] for component in nx.connected_components(tmp) if len(component) == 1)
	del tmp
	adjusted_deletions = []
	for u, v in deletions:
		if u in singletons and v in singletons:
			cost -= graph.get_edge(u, v)
			singletons.discard(u)
			singletons.discard(v)
		else:
			adjusted_deletions.append( (u, v) )
	return (adjusted_deletions, cost)

def _wtp(graph, wtpfunc, postprocessing=True):
	"""
	Compute solution for transitive projection using the weighted
	transitivity improvement heuristic. Input graph must be of type
	WeightedGraph. Returns a tuple (deletions, cost) where deletions is
	an edge list. wtpfunc() gets called for each subgraph.
	"""
	deletions = []
	cost = 0.0
	for g in nx.connected_component_subgraphs(graph):
		d, c = wtpfunc(g)
		if postprocessing:
			d, c = postprocess(g, d, c)
		deletions.extend(d)
		cost += c
	return (deletions, cost)

# helpers. those get run for each component
def _uncached_wtp_func(graph, scorefunc, subtractweight, wfactor):
	return _uncached_wtp_connected(graph, lambda g: weighted_find_culprit(g, scorefunc, subtractweight), wfactor)

def _cached_wtp_func(graph):
	dftgraph = DftGraph(graph)
	return _cached_wtp_connected(dftgraph)

# official functions. use these instead of any of the above.
def uncached_wtp(graph, postprocessing, scorefunc, subtractweight, wfactor):
	return _wtp(graph, lambda g: _uncached_wtp_func(g, scorefunc, subtractweight, wfactor), postprocessing)

def cached_wtp(graph, postprocessing=True):
	return _wtp(graph, _cached_wtp_func, postprocessing)

###################################################

def write_costlist(graph, deletions, filename):
	g = graph.copy()
	f = file(filename, "w")
	t = closure_cost(g)
# 	print >>f, "-", t
	d = 0
	delcosts = []
	for e in deletions:
		g.delete_edge(e)
		d += 1
		cost = closure_cost(g) + d
		delcosts.append(cost)
# 		print >>f, e, cost
	f.close()

	#pylab.plot(delcosts)
	#pylab.show()


def exact(graph):
	"""
	Solves Cluster Editing exactly. This function tries all 2**n
	possible edge deletions: Each edge can be either deleted or
	not. For n edges, there are 2**n ways to do this. A Gray code
	is used to make sure that in every step only one change to
	the graph needs to be done (either adding or deleting an edge).
	After each change, the cost for the transitive closure is
	computed and added to the number of deletions done so far to
	get the actual cost.

	A tuple (edges, cost) is returned, where cost is the optimal
	cost and edges is the list of edges to be removed for an
	optimal solution. The additional edges are implied by the
	transitive closure.
	"""
	g = graph.copy()
	n = g.number_of_edges()
	edges = g.edges()
	assert n < 28
	bitfield = [1]*n # 1 means that the edge is there
	best = []
	minimum = closure_cost(g)
#	print "minimum:", minimum
	dels = 0
	ic = 0
	for change in gray.grayseq_iter(n):
		if ic % 10000 == 0:
			print "%d iterations" % ic
		ic += 1
		if bitfield[change] == 1:
			assert g.has_edge(edges[change])
			g.delete_edge(edges[change])
			dels += 1
		else:
			assert not g.has_edge(edges[change])
			g.add_edge(edges[change])
			dels -= 1
		assert 0 <= dels <= n
		bitfield[change] ^= 1 # flip
		assert bitfield[change] in [0,1]
		if __debug__:
			k = len([v for v in bitfield if v])
			assert n-k == dels
		if dels >= minimum:
			continue
		cost = closure_cost(g) + dels
		if cost <= minimum:
			minimum = cost
			best = [ edges[i] for i in range(n) if bitfield[i] == 0]
#			print "minimum:",minimum, best, len(best)

	return (best, minimum)


def exact2(graph):
	"""
	Solves Cluster Editing exactly. This function tries all 2**n
	possible edge deletions: Each edge can be either deleted or
	not. For n edges, there are 2**n ways to do this. A variant of
	a Gray code is used to make sure that in every step at most two
	changes to the graph need to be done (either adding or
	deleting an edge).
	After each change, the cost for the transitive closure is
	computed and added to the number of deletions done so far to
	get the actual cost.

	A tuple (edges, cost) is returned, where cost is the optimal
	cost and edges is the list of edges to be removed for an
	optimal solution. The additional edges are implied by the
	transitive closure.
	"""
	g = graph.copy()
	n = g.number_of_edges()
	edges = g.edges()
	assert n < 50
	# note inverse logic:
	# 1 means that the edge has been deleted
	# (it's simpler that way)
	bitfield = [0]*n
	best = []
	minimum = closure_cost(g)
	for k in range(1, n):
		# A property of the way bitstrings_seq_k generates
		# the indices is that at each transition from one
		# block containing k ones to the block containing
		# k+1 ones, always the rightmost bit is set to one.
		if __debug__:
			print "n:", n, "k:",k
#		print "edges:", edges
# 		print "g.edges:", g.edges()
#		print "bitfield:", bitfield
		assert bitfield[-1] == 0
		bitfield[-1] = 1
		assert g.has_edge(edges[-1])
		g.delete_edge(edges[-1])

		cost = closure_cost(g) + k
		if cost <= minimum:
			minimum = cost
			best = [ edges[i] for i in range(n) if bitfield[i] ]

		for (i0, i1) in bitsgen.bitstrings_seq_k(n, k):
# 			print "i0:",i0,"i1:",i1
# 			print "g.edges:", g.edges()
#			print "n/k: %d/%2d %2d->%2d bitfield:" %(n,k,i0, i1), bitfield
# 			if ic > 20000:
# 				break
# 			if ic % 10000 == 0:
# 				print "%d iterations" % ic
# 			ic += 1
			# i0 is the addition, i1 the deletion
			assert not g.has_edge(edges[i0])
			g.add_edge(edges[i0])
			assert g.has_edge(edges[i1])
			g.delete_edge(edges[i1])
			assert g.number_of_edges() == n-k
			assert bitfield[i0] == 1 and bitfield[i1] == 0
			bitfield[i0] = 0
			bitfield[i1] = 1
			assert sum(bitfield) == k
			cost = closure_cost(g) + k
			if cost <= minimum:
				minimum = cost
				best = [ edges[i] for i in range(n) if bitfield[i] ]
#				print "minimum:",minimum, best, len(best)
		# this is the whole point of the exercise:
		# if k is getting too large, we can bail out early
		if k >= minimum:
			if __debug__:
				print "BAIL OUT EARLY"
			break
	return (best, minimum)

def correctly_clustered(graph, deletions, clustering):
	"""
	Check if deleting all graph edges that are in the deletions list results
	in a clustering that corresponds to the given clustering.
	Singletons are ignored.
	"""
	g = graph.copy()
	for d in deletions:
		g.delete_edge(d)
	components = nx.connected_components(g)
	components = [ sorted(component) for component in components ]
	clustering = sorted(clustering, key=len, reverse=True)
	# sort each cluster, throw out ignore singletons
	clustering = [ sorted(nodes) for nodes in clustering ]
	return components == clustering


def check_something():
	random.seed(12)
	for i in range(200000):
		print "==================== test no.", i
		n = random.randint(5, 80)
		g, changes = graphgen.perturbed_graph(graphgen.random_clustering(n, 2+n//8), 5000)
		print "%d changes" % (changes)
		deleted, cost = dft1(g)
		print "cost:", cost
		print "deleted:", deleted, len(deleted)
		if cost != changes:
			nx.write_dot(g, "prob.dot")
			os.system("neato -Tpng prob.dot > prob.png")
		assert cost == changes

def exactmain(func=exact):
	#random.seed(10)
	n = 7
	g, changes = graphgen.perturbed_graph(graphgen.random_clustering(n, 2+n//8), 5000)
	deleted_edges, cost = func(g)
	#write_costlist(g, deleted_edges, "delkosten.txt")
	print "deleted_edges:",deleted_edges, len(deleted_edges)
	print "minimum cost:", cost
	print "actual changes:", changes
	for edge in deleted_edges:
		g.delete_edge(edge)

	#draw_graph(g, deleted_edges)

def main():
#	random.seed(12); g = create_graph_with_clusters(80, 7, 10, 0.7)
#	random.seed(10); g = create_graph_with_clusters(80, 7, 25, 0.7)
	random.seed(10); g = create_graph_with_clusters(180, 10, 50, 0.75)

#	g = create_graph_with_clusters(80, 8, 50, 0.6)
#	random.seed(11); g = create_graph_with_clusters(20, 3, 5, 0.7)
#	nx.write_edgelist(g, "example04.ncol", delimiter='\t')#; return

	gn = g.copy()
	deleted_edges, cost = dft1(gn)

	write_costlist(g, deleted_edges, "delkosten.txt")
	print "deleted_edges:",deleted_edges, len(deleted_edges)
	print "minimum cost:", cost
	for edge in deleted_edges:
		gn.delete_edge(edge)

	draw_graph(g, deleted_edges)

def runstuff(iterations=10):
	for i in xrange(iterations):
		n = random.randint(5, 80)
		g = graphgen.create_graph_with_clusters(graphgen.random_clustering(n, 2+n//8), random.randint(4, 100), 0.7) # TODO
		print g.number_of_nodes(), "nodes and", g.number_of_edges(), "edges"
		deletions3, cost3 = even_better_dft(g)

def profile():
	import profile; profile.run('runstuff(5)', 'transitive.prof')

if __name__ == "__main__":
	#random.seed(1); unittest.main()
	#check_something()
	#check_exact(repeats=100, func=exact2)
	#profile()
	#main()
	#main()
	weighted_compare_dft_implementations()
