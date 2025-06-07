#! /usr/bin/env python

import networkx as nx
from weightedgraph import CompleteGraph

class ParseError(Exception):
	pass

def read_components(f, threshold, ignore_names=False):
	"""
	Returns a list of graphs read from f. Edges that
	have a weight below or equal 'threshold' will
	be ignored.
	Set ignore_names to True to use integers for the
	vertices instead of the actual (protein) names.

	File format:
	components_file ::= (component NEWLINE)* component
	component ::= (id1 id2 weight NEWLINE)+
	(id1: string, id2: string, weight: float)
	"""
	if ignore_names:
		# maps names to unique IDs
		names = {}
		unique_id = 0
	graphs = []
	g = nx.Graph()
	for l in f:
		l = l.rstrip()
		if l == "":
			# new graph
			if g.number_of_nodes() > 0:
				graphs.append(g)
			g = nx.Graph()
			continue
		fields = l.split()
		if len(fields) != 3:
			print "fields:", fields
			raise ParseError, "expecting 3 fields per line of input data"
		id1 = fields[0]
		id2 = fields[1]
		weight = float(fields[2])
		if weight >= threshold:
			if ignore_names:
				if id1 in names:
					i = names[id1]
				else:
					i = unique_id
					names[id1] = unique_id
					unique_id += 1
				if id2 in names:
					j = names[id2]
				else:
					j = unique_id
					names[id2] = unique_id
					unique_id += 1
				g.add_edge(i, j)
			else:
				g.add_edge(id1, id2)
	if g.number_of_nodes() > 0:
		graphs.append(g)
	return graphs


def weighted_read_components(f, ignore_names=False):
	"""
	Returns a list of graphs read from f.
	Set ignore_names to True to use integers for the
	vertices instead of the actual (protein) names.

	File format:
	components_file ::= (component NEWLINE)* component
	component ::= (id1 id2 weight NEWLINE)+
	(id1: string, id2: string, weight: float)
	"""
	if ignore_names:
		# maps names to unique IDs
		names = {}
		unique_id = 0
	graphs = []
	g = nx.XGraph()
	for l in f:
		l = l.rstrip()
		if l == "":
			# new graph
			if g.number_of_nodes() > 0:
				graphs.append(g)
			g = nx.XGraph()
			continue
		fields = l.split()
		if len(fields) != 3:
			print "fields:", fields
			raise ParseError, "expecting 3 fields per line of input data"
		id1 = fields[0]
		id2 = fields[1]
		weight = float(fields[2])
		if ignore_names:
			if id1 in names:
				i = names[id1]
			else:
				i = unique_id
				names[id1] = unique_id
				unique_id += 1
			if id2 in names:
				j = names[id2]
			else:
				j = unique_id
				names[id2] = unique_id
				unique_id += 1
			g.add_edge(i, j, w)
		else:
			g.add_edge(id1, id2, w)
	if g.number_of_nodes() > 0:
		graphs.append(g)
	return graphs

def read_cm(f, create_using=None):
	"""
	Read 'component matrix' file.
	File format is:
	- First line is an integer n: the number of vertices
	- Then a list of n vertex names, one per line
	- Then n-1 lines. first has n-1 tab-separated fields with edge weights,
	  second has n-2, and so on. There will be only one entry in the last
	  line.

	In this version the vertex names are ignored.

	Returns a CompleteGraph.
	"""
	n = int(f.readline().rstrip())

	# skip vertex names
	for i in xrange(n):
		f.readline()

	# read all edges but don't store them in the graph, yet
	edges = []
	for i in xrange(n-1):
		l = f.readline()
		if not l.endswith('\n'):
			raise ParseError, "file ended prematurely"
		fields = l[:-1].split('\t')
		if len(fields) != n-1-i:
			raise ParseError, "expected %d fields, got %d" % (n-1-i, len(fields))
		for j in xrange(n-1-i):
			w = float(fields[j])
			edges.append( (i, i+j+1, w) )

	# The minimum edge weight is usually the one that occurs most often.
	# We hopefully get some speed improvements by not putting those edges
	# into the graph and using the default 'missingedgecost' instead.
	if len(edges) > 0:
		min_w = min(w for (u,v,w) in edges)
	else:
		min_w = 0
#	if min_w >= 0.0:
#		raise ValueError, "Don't know what to do: Minimum edge weight is not < 0"

	g = CompleteGraph(min_w)
	g.name = "graph read from %s" % f.name
	for u, v, w in edges:
		if w != min_w:
			g.add_edge(u, v, w)
	return g

def read_cm_matrix(f):
	"""
	Read 'component matrix' file.
	File format is:
	- First line is an integer n: the number of vertices
	- Then a list of n vertex names, one per line
	- Then n-1 lines. first has n-1 tab-separated fields with edge weights,
	  second has n-2, and so on. There will be only one entry in the last
	  line.

	Vertex names are ignored.

	Returns a MatrixGraph.
	"""

	from matrixgraph import MatrixGraph

	n = int(f.readline().rstrip())

	# skip vertex names
	for i in xrange(n):
		f.readline()

	g = MatrixGraph(n)
	edges = []
	for i in xrange(n-1):
		l = f.readline()
		if not l.endswith('\n'):
			raise ParseError, "file ended prematurely"
		fields = l[:-1].split('\t')
		if len(fields) != n-1-i:
			raise ParseError, "expected %d fields, got %d" % (n-1-i, len(fields))
		for j in xrange(n-1-i):
			w = float(fields[j])
			g.add_edge(i, i+j+1, w)

	g.name = "graph read from %s" % f.name

	return g

def read_blast(f, threshold, ignore_names=False):
	"""
	Reads in a BLAST result file and returns a list of graphs.
	- There must be an edge for both directions (id1 => id2 and id2 => id1).
	  The weight of the resulting edge will be -log(e-value), where e-value is
	  the worse of the two values.
	"""

	d = {}
	ok = []
	for l in sys.stdin:
		l = l.rstrip()
		fields = l.split()
		if len(fields) != 12:
			raise DataError, "expecting 12 fields per line of input data"
		u = fields[0]
		v = fields[1]
		weight = float(fields[10])
		if weight <= threshold and u != v:
			key = (v, u)
			if key in d:
				# edge in other direction already encountered
				d[key] = min(d[key], weight)
				ok.append(key)
			else:
				d[(u, v)] = weight

	for u,v in ok:
		w = d[(u,v)]
		if w != 0:
			x = -math.log10(w)
		else:
			x = 310
		print "%s\t%s\t%.2f" % (u, v, x)

def write_cm(graph, f):
	"""
	Writes 'component matrix' file.
	f must be a file name or a file open for writing.
	See read_cm for the file format. For nonexisting edges a -1 will be written.
	"""
	if type(f) is str:
		name = f
		f = file(name, "w")
	nodes = graph.nodes()
	print >>f, len(nodes)
	for u in nodes:
		print >>f, "V%d" % u

	if hasattr(graph, "get_edge"):
		for i, u in enumerate(nodes):
			weights = []
			for j in xrange(i+1, len(nodes)):
				weights.append(graph.get_edge(u, nodes[j]))
			print >>f, '\t'.join(map(str, weights))
	else:
		for i, u in enumerate(nodes):
			weights = []
			for j in xrange(i+1, len(nodes)):
				if graph.has_edge(u, nodes[j]):
					weights.append(1)
				else:
					weights.append(-1)
			print >>f, '\t'.join(map(str, weights))


class FileFormatError(Exception):
	pass

def read_spiders(f):
	"""Read spider-vs-tree file
	Format:
	Tree1 \t Tree2 \t ... \t Tree-n
	Spider1 \t value1 \t ... \t value-n
	...
	Spider-m \t value \t ... \t value-n

	In other words: A matrix of tab-separated fields, one line per spider,
	first line names the trees and has one field less than the others.

	Returns (data, trees, spiders) tuple, where data is the matrix,
	trees are the names of the trees and spiders are the names of the
	spiders.
	"""
	line = f.readline().rstrip()
	trees = line.split("\t")
	n = len(trees) # expected number of data fields in the following lines
	data = []
	spiders = [] # names of spiders
	for line in f:
		fields = line.split("\t")
		if len(fields) != n+1:
			raise FileFormatError, "expected %d fields, got %d" % (n+1, len(fields))
		spiders.append(fields[0])
		data.append(map(int, fields[1:]))
	return (data, trees, spiders)
