#!/usr/bin/env python
import unittest

def gray(n):
	"""Returns gray code for n bits"""
	if n == 0:
		return []
	if n == 1:
		return [[0],[1]]
	seq = []
	for s in gray(n-1):
		s.append(0)
		seq.append(s)
	for s in reversed(gray(n-1)):
		s.append(1)
		seq.append(s)
	return seq


def grayseq(n):
	"""
	Returns list of bit indices: Start with [0]*n and flip the bits in this order
	to get the gray code as in the function gray() as above.
	"""
	if n == 0:
		return []
	seq = [0]
	for i in range(1, n):
		seq = seq + [i] + seq
	return seq

def grayseq_iter(n):
	if n == 0:
		return
	for v in grayseq_iter(n-1):
		yield v
	yield n-1
	for v in grayseq_iter(n-1):
		yield v
	return

def _graybyseq(n):
	"""
	Just for testing. Use list of bit change indices from grayseq() to
	get a gray code sequence.
	"""
	if n == 0:
		return []
	c = [0]*n
	seq = [c[:]]
	for change in grayseq(n):
		c[change] ^= 1 # bitwise XOR
		seq.append(c[:])
	return seq


class TestGray(unittest.TestCase):
	def test(self):
		for i in range(12):
			self.assertEqual(_graybyseq(i), gray(i))
	
	def testIter(self):
		for i in range(12):
			self.assertEqual(grayseq(i), list(grayseq_iter(i)))


if __name__ == "__main__":
	unittest.main()
