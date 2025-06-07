#!/usr/bin/env python

# TODO: generator
"""
Generating all bit vectors of length n. The vectors
are ordered by number of ones and fulfill the
condition that one vector differs from the next in
the sequence in most two positions.

Example sequence for n=3:
0 0 0
0 0 1
1 0 0
0 1 0
0 1 1
1 0 1
1 1 0
1 1 1
"""

def bitstrings_k(n, k):
	assert k <= n
	if n == 0:
		return [ [] ]
	if n == 1:
		if k == 0:
			return [ [0] ]
		else:
			return [ [1] ]
	# special case
	if k == 0:
		return [ [0]*n ]
	if k == n:
		return [ [1]*n ]

	t1 = bitstrings_k(n-1, k-1)
	t0 = bitstrings_k(n-1, k)
	for s in t1:
		s.append(1)
	for s in t0:
		s.append(0)

	t0.reverse()
	s = t1 + t0

	return s


def bitstrings(n):
	s = []
	for k in xrange(n+1):
		s.extend(bitstrings_k(n, k))
	return s


def _bitstrings_seq(n):
	"""
	Uses bitstrings(n) to generate a sequence of index tuples denoting the index,
	at which a bit has to be unset or set. Each element of the sequence is a
	tuple (i0, i1) where i1 is the index of the bit to set to zero and
	i1 the index of the bit to set to one. i0 may be None.
	"""
	s = []
	bits = bitstrings(n)
	prev = bits[0]
	for j in range(1, len(bits)):
		b = bits[j]
		if __debug__:
			i1 = None
		i0 = None
		for i in range(len(b)):
			if not b[i] == prev[i]:
				if b[i] == 0:
					i0 = i
				else:
					i1 = i
		assert i0 != -1 # make sure it's set
		assert i1 is not None
		s.append( (i0, i1) )
		prev = b
	return s

def _bitstrings_seq_k(n, k):
	"""
	Uses bitstrings_k(n,k) to generate a sequence of index tuples denoting the index,
	at which a bit has to be unset or set. Each element of the sequence is a
	tuple (i0, i1) where i1 is the index of the bit to set to zero and
	i1 the index of the bit to set to one. i0 may be None.
	"""
	s = []
	bits = bitstrings_k(n, k)
	prev = bits[0]
	for j in range(1, len(bits)):
		b = bits[j]
		i1 = None
		i0 = None
		for i in range(len(b)):
			if not b[i] == prev[i]:
				if b[i] == 0:
					i0 = i
				else:
					i1 = i
		assert i0 != -1 # make sure it's set
		assert i1 is not None and i0 is not None
		s.append( (i0, i1) )
		prev = b
	return s


def _bitstrings_seq2(n, k): # TODO test
	s = []
	for k in xrange(n):
		s.extend(_bitstrings_seq_k(n, k))
		s.append( (None, n-1) )
	return s


def bitstrings_seq_k(n, k):
	"""

	"""
	assert 0 <= k <= n
	global _cache, _cache_max
	if (n,k) in _cache:
		return _cache[(n,k)]
	if k == 0 or k == n:
		return []
	# assemble sequence from parts
	s = bitstrings_seq_k(n-1, k-1)[:]
	if k == n-1:
		s.append( (n-1, n-2) )
	else:
		s.append( (n-1, n-k-2) )
	s1 = bitstrings_seq_k(n-1, k)
	s1 = [ (y,x) for (x,y) in s1 ]
	s.extend(reversed(s1))
	if n <= _cache_max:
		_cache[(n,k)] = s
	return s


def bitstrings_seq_k_iter(n, k):
	if k == 0 or k == n:
		return

	for v in bitstrings_seq_k_iter(n-1, k-1):
		yield v
	if k == n-1:
		yield (n-1, n-2)
	else:
		yield (n-1, n-k-2)
	for x,y in reversed(list(bitstrings_seq_k_iter(n-1, k))): # TODO avoid list()
		yield (y,x)


def bitstrings_seq(n):
	"""
	Returns a list of index pairs (i0, i1). Start out from a bit vector
	[0]*n and at each step, unset the bit at i0 and set the bit at i1. The
	sequence of bit vectors that you get by this process is the same as
	the one you would have received by calling bitstrings(n).
	"""
	s = []
	for k in xrange(n):
		s.extend(bitstrings_seq_k(n, k))
		s.append( (None, n-1) )
	return s


def bitstrings_seq_iter(n):
	for k in xrange(n):
		for v in bitstrings_seq_k_iter(n, k):
			yield v
		yield (None, n-1)



def generate_pbm(n):
	"""
	Writes a PBM (portable bitmap) for bitstrings(n) to stdout.
	"""
	print "P1"
	print "# n=%d" % n
	print n,2**n
	#m = ['0', '1']
	for b in bitstrings(n):
		s = ' '.join([ `c` for c in b])
		print s

def nice_print(n):
	print "# n =", n
	for k in range(n+1):
		for b in bitstrings_k(n,k):
			s = ' '.join([ `c` for c in b])
			print s
		print


_cache_max = 12
_cache = {}
for i in range(_cache_max+1):
	_cache[(i, 0)] = []
	_cache[(i, i)] = []

