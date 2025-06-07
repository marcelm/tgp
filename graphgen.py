#! /usr/bin/env python

"""
Generators for various kinds of graphs.
"""

import networkx as nx
import unittest
import random

def fac(n):
	if n == 0 or n == 1:
		return 1
	else:
		return n*fac(n-1)

def choose(n, k):
	x = 1
	for i in range(n-k+1, n+1):
		x *= i
	x /= fac(k)
	return x

def random_partition(end, n):
	"""
	Partitions the interval [0, end[ into n intervals of random
	(but integer) length. end must be integer.
	A list of tuples (begin, end) representing the intervals
	[begin, end[ is returned. There will be no empty intervals.

	For example, random_partition(20, 3)
	might return [(0, 7), (7, 12), (12, 20)].
	"""
	samples = random.sample(xrange(1, end), n-1)
	samples.sort()

	intervals = []
	prev = 0
	for r in samples:
		intervals.append( (prev, r) )
		prev = r
	intervals.append( (prev, end) )
	return intervals

class TestRandomPartition(unittest.TestCase):
	def test(self):
		for i in range(1000):
			end = random.randint(0, 1500)
			n = end / 10 + 1
			partition = random_partition(end, n)
			self.assertEqual(n, len(partition))
			self.assertEqual(0, partition[0][0])
			self.assertEqual(end, partition[-1][1])
			for j in range(n-1):
				self.assertEqual(partition[j][1], partition[j+1][0])

def random_clustering(n, k):
	"""
	Create a random clustering of n nodes into k clusters
	Returns a list of clusters, where a cluster is simply a list of nodes.
	For example, three clusters of size 4, 3, and 1:
	[ [0, 1, 2, 3], [4, 5, 6], [7] ]
	Because random_partition is used, no cluster will be empty.
	"""
	s = []
	for interval in random_partition(n, k):
		s.append(range(interval[0], interval[1]))
	return s

def create_graph_with_clusters(clustering, nadditional, edges_fraction=0.8, weighted=False):
	"""
	Creates a random graph with nnodes nodes and nclusters "clusters".
	If weighted is True, the returned graph will be of type XGraph and all
	edges will have random weights between -1 and +1.
	Otherwise the returned graph is of type Graph.
	"""
	if weighted:
		g = nx.XGraph()
	else:
		g = nx.Graph()
	for nodes in clustering:
		# go over all possible edges (no loops)
		for i in range(len(nodes)):
			for j in range(i+1, len(nodes)):
				if random.random() < edges_fraction:
					if weighted:
						g.add_edge(nodes[i], nodes[j], random.random()*2 - 1)
					else:
						g.add_edge(nodes[i], nodes[j])

	# now the additional edges between clusters
	for i in xrange(nadditional):
		cluster1, cluster2 = random.sample(clustering, 2)
		n1 = random.choice(cluster1)
		n2 = random.choice(cluster2)
		if weighted:
			g.add_edge(n1, n2, random.random()*2 - 1)
		else:
			g.add_edge(n1, n2)
	return g


def completely_random_graph(n, prob):
	"""
	Create a random unweighted graph on n nodes where the probability that
	an edge exists is prob.
	"""
	g = nx.Graph()
	for i in xrange(n):
		for j in xrange(i, n):
			if random.random() < prob:
				g.add_edge(i, j)
	return g

def perturbed_graph(clustering, changes):
	"""
	Creates a random graph with the given clustering (list of lists of nodes).
	The edges of the graph will be modified 'changes' times. A modification
	means that an edge is deleted if it is already there or added if it is not.
	Care is taken to not insert an edge if it has been deleted and vice-versa.
	It may be that the number of changes requested under this constraint
	is too high, therefore a tuple (graph, changes) is returned which contains
	the graph and the actual number of changes performed.
	"""
	g = nx.Graph()
	nnodes = sum(map(len, clustering))

	# maps a node to the number of "perturbations" still allowed for it.
	# A perturbation is either removal of the edge to a neigbor or adding
	# a new edge to some other node.
	permitted_deletions = [0] * nnodes
	permitted_additions = [0] * nnodes
	for nodes in clustering:
		assert len(nodes) > 0
		if len(nodes) == 1:
			# special case for singletons
			g.add_node(nodes[0])
			continue
		# go over all possible edges (no loops)
		# how often a node may be perturbed depends on the number of
		# nodes in its cluster
		perturb_del = (len(nodes) - 1) // 4 # "less than n/4 deletions"
		perturb_add = (len(nodes) - 1) // 4 # "less than n/4 additions"
		for i in range(len(nodes)):
			permitted_deletions[nodes[i]] = perturb_del
			permitted_additions[nodes[i]] = perturb_add
			for j in range(i+1, len(nodes)):
				g.add_edge(nodes[i], nodes[j])

	changed = {}
	retry_counter = 0
	c = 0
	while c < changes:
		# pick two nodes at random
		i, j = random.sample(xrange(nnodes), 2)
#		print "picked", (i,j)
		if (i,j) in changed or (j,i) in changed:
			# pick again
			continue
		if g.has_edge(i, j):
			if permitted_deletions[i] > 0 and permitted_deletions[j] > 0:
				g.delete_edge(i, j)
				permitted_deletions[i] -= 1
				permitted_deletions[j] -= 1
#				print "CHANGE: del", (i,j)
				changed[(i,j)] = None
				retry_counter = 0
				c += 1
			else:
				retry_counter += 1
		else:
			if permitted_additions[i] > 0 and permitted_additions[j] > 0:
				g.add_edge(i, j)
				permitted_additions[i] -= 1
				permitted_additions[j] -= 1
#				print "CHANGE: add", (i,j)
				changed[(i,j)] = None
				retry_counter = 0
				c += 1
			else:
				retry_counter += 1
		# TODO this could be done a bit smarter
		if retry_counter == 100:
			break
	assert len(changed) == c
	return (g, c)

def bad_graph(N1, N2):
	"""
	"""
	#N1 = 33
	#N2 = 5
	clustering = [ range(N1), range(N1, N1+N2), range(N1+N2, N1+N2+N2) ]
	g = nx.Graph()
	nnodes = sum(map(len, clustering))

	# maps a node to the number of "perturbations" still allowed for it.
	# A perturbation is either removal of the edge to a neigbor or adding
	# a new edge to some other node.
	permitted_deletions = [0] * nnodes
	permitted_additions = [0] * nnodes
	for nodes in clustering:
		assert len(nodes) > 0
		if len(nodes) == 1:
			# special case for singletons
			g.add_node(nodes[0])
			continue
		# go over all possible edges (no loops)
		# how often a node may be perturbed depends on the number of
		# nodes in its cluster
		perturb_del = (len(nodes) - 1) // 4 # "less than n/4 deletions"
		perturb_add = (len(nodes) - 1) // 4 # "less than n/4 additions"
		for i in range(len(nodes)):
			permitted_deletions[nodes[i]] = perturb_del
			permitted_additions[nodes[i]] = perturb_add
			for j in range(i+1, len(nodes)):
				g.add_edge(nodes[i], nodes[j])

	pd = permitted_deletions[0]
	pa = permitted_additions[0]
	ops = 0
	for j in xrange(2, 2+pd):
		assert g.has_edge(0, j)
		g.delete_edge(0, j)
		ops += 1
	for j in xrange(2+pd, 2+pd+pd):
		assert g.has_edge(1, j)
		g.delete_edge(1, j)
		ops += 1

	for j in xrange(N1, N1+pa):
		assert not g.has_edge(0, j)
		g.add_edge(0, j)
		ops += 1
	for j in xrange(N1+N2, N1+N2+pa):
		assert not g.has_edge(1, j)
		g.add_edge(1, j)
		ops += 1

	pd1 = permitted_deletions[N1]
	for j in xrange(N1+pa, N1+pa+pd1):
		if g.has_edge(N1, j):
			g.delete_edge(N1, j)
			ops += 1
	#assert ops == 2*pd+2*pa+pd1
	print "ops:", ops
	return (g, ops)#2*pd+2*pa+pd1)
	
	# pick three clusters
	a, b, c = random.sample(clustering, 3)

	# pick an edge from the first cluster
	u, v = random.sample(a, 2)

	# pick a node from the second cluster
	w = random.choice(b)

	tries = 0
	while permitted_additions[u] > 0:
		x = random.choice(c)




	changed = {}
	retry_counter = 0
	c = 0
	while c < changes:
		# pick two nodes at random
		i, j = random.sample(xrange(nnodes), 2)
#		print "picked", (i,j)
		if (i,j) in changed or (j,i) in changed:
			# pick again
			continue
		if g.has_edge(i, j):
			if permitted_deletions[i] > 0 and permitted_deletions[j] > 0:
				g.delete_edge(i, j)
				permitted_deletions[i] -= 1
				permitted_deletions[j] -= 1
#				print "CHANGE: del", (i,j)
				changed[(i,j)] = None
				retry_counter = 0
				c += 1
			else:
				retry_counter += 1
		else:
			if permitted_additions[i] > 0 and permitted_additions[j] > 0:
				g.add_edge(i, j)
				permitted_additions[i] -= 1
				permitted_additions[j] -= 1
#				print "CHANGE: add", (i,j)
				changed[(i,j)] = None
				retry_counter = 0
				c += 1
			else:
				retry_counter += 1
		# TODO this could be done a bit smarter
		if retry_counter == 100:
			break
	assert len(changed) == c
	return (g, c)
