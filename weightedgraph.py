"""
An implementation of NetworkX's Graph with weighted edges.
Edges have the special property that they are considered to
not exist when their weight is nonpositive. This
applies only to the methods
 - G.edges_iter(), G.edges_iter(nbunch)
 - G.edges()
 - G.neighbors(v), G[v]
 - G.neighbors_iter(v)
but NOT, for example, to
 - G.has_neighbor(n1, n2)
 - G.get_edge(n1, n2)

TODO The methods degree() and degree_iter() have not
been implemented.

This class was designed such that the connected component methods
(connected_components, connected_component_subgraphs)
*ignore* the negative-weight edges.

Notes:

connected_components uses:
- single_source_shortest_path
- single_source_shortest_path_length

those use:
- neighbors, which uses neighbors_iter, which uses edges_iter
"""

from networkx import NetworkXError, Graph


class WeightedGraph(Graph):
	def __init__(self, data=None, **kwds):
		"""Initialize WeightedGraph.
		For the data parameter, see Graph.__init__.
		"""
		if kwds.get("selfloops", False) or kwds.get("multiedges", False):
			# before this is enabled, someone needs to think about whether it works
			raise NetworkXError, "neither selfloops nor multiedges may be set"

		Graph.__init__(self, data, **kwds)

	def neighbors_iter(self, n):
		for v, w in self.adj[n].iteritems():
			if w > 0.0:
				yield v

	def __getitem__(self, key):
		"Used by connected_components (indirectly)"
		adj = self.adj[key]
		return {n: w for n, w in adj.items() if w > 0.0}

	def edges_iter(self, nbunch=None, data=None):
		"""
		Returns iterator over those edges adjacent to nodes in nbunch that have
		nonnegative weight.
		If nbunch is None, iterates over all nonnegative-weight edges.
		"""
		for u, v, w in Graph.edges_iter(self, nbunch, data=True):
			if w > 0.0:
				if data:
					yield (u,v,w)
				else:
						yield (u, v)

	def all_edges(self, nbunch=None):
		"""
		Returns list of all edges, also those with negative weight.
		"""
		return list(self.all_edges_iter(nbunch))

	all_edges_iter = Graph.edges_iter

	def number_of_edges(self):
		c = 0
		for u,d in self.adj.iteritems():
			c += sum(1 for v,w in d.iteritems() if w > 0.0)
		return c/2

	number_of_all_edges = Graph.number_of_edges

	def copy(self):
		"""Returns copy of the graph"""
		graph = self.__class__()
		graph.name = self.name
		if hasattr(self, "dna"):
			graph.dna = self.dna.copy()
		for n in self:
			graph.add_node(n)
		for u, v, w in self.all_edges_iter():
			graph.add_edge(u, v, w)
		return graph

class CompleteGraph(WeightedGraph):
	def __init__(self, missingweight, data=None, **kwds):
		assert missingweight < 0.0
		self.missingweight = missingweight
		WeightedGraph.__init__(self, data, **kwds)

	def get_edge(self, u, v=None):
		# copied from Graph
		try:
			return self.adj[u][v]    # raises KeyError if edge not found
		except KeyError:
			return self.missingweight

	def copy(self):
		graph = self.__class__(self.missingweight)
		graph.name = self.name
		if hasattr(self, "dna"):
			graph.dna = self.dna.copy()
		for n in self:
			graph.add_node(n)
		for u, v, w in self.all_edges_iter(data=True):
			graph.add_edge(u, v, w)
		return graph

	def subgraph(self, nbunch, inplace=False, create_using=None):
		assert not inplace
		g = None
		assert not create_using
		# Copied from NetworkX
		bunch = set(self.nbunch_iter(nbunch))
		# create new graph and copy subgraph into it
		H = WeightedGraph()
		H.name = "Subgraph of (%s)"%(self.name)
		# add edges
		H_adj = H.adj # cache
		self_adj = self.adj # cache
		for n in bunch:
			H_adj[n] = dict( ((u,d) for u,d in self_adj[n].iteritems()
							if u in bunch) )

		cg = CompleteGraph(self.missingweight, data=H)
		return cg
