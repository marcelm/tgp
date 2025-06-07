"""
An implementation of NetworkX's XGraph with weighted edges.
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

from networkx import XGraph, NetworkXError

class WeightedGraph(XGraph):
	def __init__(self, data=None, **kwds):
		"""Initialize WeightedGraph.
		For the data parameter, see XGraph.__init__.
		"""
		if kwds.get("selfloops", False) or kwds.get("multiedges", False):
			# before this is enabled, someone needs to think about whether it works
			raise NetworkXError, "neither selfloops nor multiedges may be set"

		XGraph.__init__(self, data, **kwds)

	def neighbors_iter(self, n):
		for v,w in self.adj[n].iteritems():
			if w > 0.0:
				yield v

	def edges_iter(self, nbunch=None):
		"""
		Returns iterator over those edges adjacent to nodes in nbunch that have
		nonnegative weight.
		If nbunch is None, iterates over all nonnegative-weight edges.
		"""
		for u,v,w in XGraph.edges_iter(self, nbunch):
			if w > 0.0:
				yield (u,v,w)

	def all_edges(self, nbunch=None):
		"""
		Returns list of all edges, also those with negative weight.
		"""
		return list(self.all_edges_iter(nbunch))

	all_edges_iter = XGraph.edges_iter

	def number_of_edges(self):
		c = 0
		for u,d in self.adj.iteritems():
			c += sum(1 for v,w in d.iteritems() if w > 0.0)
		return c/2

	number_of_all_edges = XGraph.number_of_edges

	def copy(self):
		"""Returns copy of the graph"""
		graph = self.__class__()
		graph.name = self.name
		if hasattr(self, "dna"):
			graph.dna = self.dna.copy()
		for n in self:
			graph.add_node(n)
		for e in self.all_edges_iter():
			graph.add_edge(e)
		return graph

class CompleteGraph(WeightedGraph):
	def __init__(self, missingweight, data=None, **kwds):
		assert missingweight < 0.0
		self.missingweight = missingweight
		WeightedGraph.__init__(self, data, **kwds)

	def get_edge(self, u, v=None):
		# copied from XGraph
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
		for e in self.all_edges_iter():
			graph.add_edge(e)
		return graph

	def subgraph(self, nbunch, inplace=False, create_using=None):
		g = None
		if create_using is None:
			g = self.__class__(
					self.missingweight,
					multiedges=self.multiedges,
					selfloops=self.selfloops)
		return WeightedGraph.subgraph(self, nbunch, inplace, create_using=g)
