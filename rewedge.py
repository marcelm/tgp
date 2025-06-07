#!/usr/bin/env python

import sys, getopt

import graphread
import graphalgo

def usage():
	print """
Usage
 rewedge <component matrix files> ...

Options
  -M <size>  maximum size of component                (default: no limit)
  -r         disable postprocessing ("raw")           (default: enabled)
  -u         use the uncached algorithm               (default: cached)
  -s <scorefunc> score function to use:
             min, min2, sum, sumabs, or missing       (default: min)
  -w         DON'T subtract weight in score funct.    (default: do subtract)
  -f <fact>  edge removal factor: s'(uv)=fact*s(uv)
             special value "-inf" is recognized       (default: -1)
  -d         print deletions                          (default: don't)
  -v         verbose                                  (default: off)
"""

def normalized(deletions):
	for u, v in deletions:
		if u > v:
			yield (v, u)
		else:
			yield (u, v)

def main():
	try:
		opts,args = getopt.getopt(sys.argv[1:], "M:vrus:wf:d", ("profile",))
	except getopt.GetoptError:
		usage()
		sys.exit(1)
	max_size = None
	verbose = False
	postprocessing = True
	cached = True
	wfactor = -1
	subtractweight = True
	print_deletions = False
	scorefunc = graphalgo.score_min
	scorefunc_changed = False
	subtract_changed = False
	scorefuncs = {
		'min': graphalgo.score_min,
		'min2': graphalgo.score_min2,
		'sum': graphalgo.score_sum,
		'sumabs': graphalgo.score_sumabs,
		'missing': graphalgo.score_missing }
	for o, a in opts:
		if o == '-M':
			max_size = int(a)
		elif o == '--profile':
			print >>sys.stderr, "profiling enabled"
		elif o == '-v':
			verbose = True
		elif o == '-r':
			postprocessing = False
		elif o == '-u':
			cached = False
		elif o == '-s':
			if a in scorefuncs:
				scorefunc = scorefuncs[a]
				scorefunc_changed = True
			else:
				print >>sys.stderr, "unrecognized score function", a
				usage()
				sys.exit()
		elif o == '-w':
			subtractweight = False
		elif o == '-f':
			if a == "-inf":
				wfactor = -1e50000
			else:
				wfactor = float(a)
		elif o == '-d':
			print_deletions = True
		if subtract_changed or scorefunc_changed and cached:
			print >>sys.stderr, """sorry, changing score function and disabling weight
subtraction not supported for the cached algorithm"""
			sys.exit(1)
	if len(args) == 0:
		print >>sys.stderr, "need component matrix (.cm) files"
		usage()
		sys.exit(1)

	print "# max component size:", max_size
	print "# <component no.>\t<n>\t<m>\t<heuristic cost>"

	if cached:
		algo = graphalgo.cached_wtp
	else:
		algo = graphalgo.uncached_wtp
	total = 0.0
	n = 0
	component_n = -1
	for fname in args:
		if verbose:
			print >>sys.stderr, "computing %s ..." % fname
		f = file(fname)
		#graph = graphread.read_cm_matrix(f)
		graph = graphread.read_cm(f)
		component_n += 1
		if max_size is not None and graph.number_of_nodes() > max_size:
			if verbose:
				print >>sys.stderr, "skipped"
			print "%d\t%d\t%d\tNA" % (component_n, graph.number_of_nodes(), graph.number_of_edges())
			continue
		sys.stdout.write("%d\t%d\t%d\t" % (component_n, graph.number_of_nodes(), graph.number_of_edges()))
		if sys.stdout.isatty:
			sys.stdout.flush()
		if algo is graphalgo.uncached_wtp:
			dels, cost = algo(graph, postprocessing, scorefunc, subtractweight, wfactor)
		else:
			dels, cost = algo(graph, postprocessing)
		#print "graph:", graph.all_edges()
		#print "dels:", dels, len(dels)
		#import networkx as nx;nx.write_dot(graph, "bla.dot")

		if __debug__:
			delcost = 0.0
			gc = graph.copy()
			for u, v in dels:
				delcost += gc.get_edge(u, v)
				gc.add_edge(u, v, 0.0)
			wcc = graphalgo.weighted_closure_cost(gc)
			#print "wcc:", wcc, "delcost:", delcost, "cost:", cost
			assert cost == wcc+delcost

		total += cost
		if print_deletions:
			print "%s\t%s" % (cost, sorted(normalized(dels)))
		else:
			print cost
		#sys.stdout.flush()
#		print "cost: %f (total: %f)\n" % (cost, total_cost_min)
		n += 1
	if verbose:
		print >>sys.stderr, "===================================="
	print "# %d of %d components computed." % (n, len(args))
	print "# total cost:", total

if __name__ == "__main__":
	if len(sys.argv) > 1 and sys.argv[1] == '--profile':
		import profile; profile.run('main()', 'rewedge.prof')
	else:
		main()
