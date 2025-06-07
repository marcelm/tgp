import networkx as nx
import md5
import os
import pickle

def cached_layout(graph, cachedir="layout-cache"):
	"""
	Returns a cached layout if the given graph is in the cache.
	If not, runs graphviz and stores the layout in the cache.
	If the cachedir doesn't exist, it will be created.
	"""
	# This calculates a hash over the graph -- hash(graph) doesn't work
	# because it yields different values on different Python runs.
	edges = graph.edges()
	if len(edges) > 0 and len(edges[0]) == 3:
		# weighted graph
		edges = [ (u,v) for u,v,w in edges ]
	edges.sort()
	m = md5.new()
	m.update(pickle.dumps(edges))
	md5digest = m.hexdigest()

	fname = os.path.join(cachedir, "%s.layout" % md5digest[:8]) # only first 8 characters
	if not os.path.isdir(cachedir):
		os.mkdir(cachedir)
	if os.path.isfile(fname):
		# found in cache
		return pickle.load(file(fname))
	else:
		pos = nx.graphviz_layout(graph)
		pickle.dump(pos, file(fname, "w"))
		return pos


def draw_graph(graph, deleted):
	"""
	Draws a graph. 'deleted' is a list of edges that will be drawn differently.
	"""
#	print "graph:", graph.edges(), graph.number_of_edges()
	# the import is here for two reasons
	try:
		import pylab
	except ImportError:
		print "pylab not found, not drawing graph"
		return

	remaining_edges = [ (n1,n2) for (n1,n2) in graph.edges_iter() if not (n1,n2) in deleted and not (n2,n1) in deleted ]
	pos = cached_layout(graph)
	labelpos = {}
	for k,v in pos.iteritems():
		labelpos[k] = (v[0], v[1]+10)
	#clustering_coefficients = nx.clustering(g, g.nodes(), with_labels=True)

	#edge_coefficients = [ 0.01 + eccs[edge] for edge in remaining_edges ]

	#print eccs
	#print edge_coefficients
	#for k,v in clustering_coefficients.iteritems():
	#	clustering_coefficients[k] = "%.2f"%v

#	nx.draw(g, pos=pos, edgelist=edges, alpha=0.5)
	pylab.gca().set_xticks([])
	pylab.gca().set_yticks([])

	# nodes and their labels
	nx.draw_networkx_nodes(graph, pos=pos)
	nx.draw_networkx_labels(graph, pos=pos)

	# regular edges
	nx.draw_networkx_edges(graph, pos=pos, edgelist=remaining_edges, alpha=0.6)

	# deleted edges
	nx.draw_networkx_edges(graph, pos=pos, edgelist=deleted, edge_color='g', width=2, style='dotted')

	#nx.draw_networkx_labels(g, pos=labelpos, labels=clustering_coefficients, font_size=8, font_color='b')
	#pylab.box(False)
	#pylab.figtext(0.5, 0.05, "blabla")
	#pylab.savefig('image.eps')
	pylab.show()


def weighted_draw_graph(graph, deleted):
	"""
	Draws a graph. 'deleted' is a list of edges that will be drawn differently.
	graph should be of type XGraph.
	"""
	print "graph:", graph.edges(), graph.number_of_edges()
	# the import is here for two reasons
	try:
		import pylab
	except ImportError:
		print "pylab not found, not drawing graph"
		return

	remaining_edges = [ (n1,n2,w) for (n1,n2,w) in graph.edges_iter() if not (n1,n2) in deleted and not (n2,n1) in deleted ]
	pos = cached_layout(graph)
	labelpos = {}
	for k,v in pos.iteritems():
		labelpos[k] = (v[0], v[1]+10)
	#clustering_coefficients = nx.clustering(g, g.nodes(), with_labels=True)

	#edge_coefficients = [ 0.01 + eccs[edge] for edge in remaining_edges ]

	#print eccs
	#print edge_coefficients
	#for k,v in clustering_coefficients.iteritems():
	#	clustering_coefficients[k] = "%.2f"%v

#	nx.draw(g, pos=pos, edgelist=edges, alpha=0.5)
	pylab.gca().set_xticks([])
	pylab.gca().set_yticks([])

	# nodes and their labels
	nx.draw_networkx_nodes(graph, pos=pos)
	nx.draw_networkx_labels(graph, pos=pos)

	weights = [ (w+1)*5 for u,v,w in remaining_edges ]
	deleted_weights = [ (graph.get_edge(u, v)+1)*5 for u,v in deleted ]
	
	# regular edges
	nx.draw_networkx_edges(graph, pos=pos, edgelist=remaining_edges, alpha=0.6, width=weights)

	# deleted edges
	nx.draw_networkx_edges(graph, pos=pos, edgelist=deleted, edge_color='g', style='dotted', width=deleted_weights)

	#nx.draw_networkx_labels(g, pos=labelpos, labels=clustering_coefficients, font_size=8, font_color='b')
	#pylab.box(False)
	#pylab.figtext(0.5, 0.05, "blabla")
	#pylab.savefig('image.eps')
	pylab.show()

def main():
	import graphalgo, graphgen, random
	random.seed(10); g = graphgen.create_graph_with_clusters(graphgen.random_clustering(70, 8), 40, 0.8, weighted=True)
	print "orig graph:", g.edges()
	threshold = 0.0
	f = graphalgo.filtered_graph(g, threshold)
#	g = create_graph_with_clusters(80, 8, 50, 0.6)
#	random.seed(11); g = create_graph_with_clusters(20, 3, 5, 0.7)
#	nx.write_edgelist(g, "example04.ncol", delimiter='\t')#; return

	#gn = f.copy()
	deleted_edges, cost = graphalgo.weighted_dft(g, threshold)

	print "deleted_edges:",deleted_edges, len(deleted_edges)
	print "minimum cost:", cost
	#for edge in deleted_edges:
	#	gn.delete_edge(edge)

	weighted_draw_graph(f, deleted_edges)


if __name__ == "__main__":
	main()
