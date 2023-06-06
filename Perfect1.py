# Copyright 2023 Jean-Marie Mirebeau, Centre Borelli, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This python file is meant to compute the set Perfect1(M0), defined in the paper  
Monotone discretization of anisotropic differential operators using Voronoi’s first reduction
by Bonnans, Bonnet, Mirebeau
"""

# Original data : 
# /Users/jean-mariemirebeau/Dropbox/Latex/2020/6_Juin/Voronoi6

import sys; sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")

import numpy as np
import itertools
from Ryskov import Ryskov_load,norm2_AV,dot_AtDA,dot,outer_self
from scipy.optimize import linprog

np.set_printoptions(linewidth=1000)


def Voronoi(D,maxiter=100):
	"""
	Compute Voronoi's reduction of the symmetric positive definite matrix D,
	i.e. the perfect form M which minimizes Tr(DM).
	We also return the pair (i,g), where i is the index of the reference perfect form, 
	and g is the linear vhange of variables for the arithmetic equivalence with M.

	Note : Computing the matrix decomposition associated with Voronoi's first reduction requires an additional step, 
	which is not implemented here. See the AGD library, which is also Much more optimized.
	"""

	# Load data, basic pre-processing
	Dold=D
	ndim = len(D)
	Ryskov_data = Ryskov_load(ndim)
	# Ms : the list of perfect forms, normalized so that minimal vectors have norm one
	Ms = np.stack([x.M/x.M[0,0] for x in Ryskov_data],axis=-1)
	# The list of neighbors of each perfect form
	neigh_Ms = [dot_AtDA(Ms[...,x.neigh_i],x.neigh_g) for x in Ryskov_data]

	# Set the starting vertex of the simplex-like method
	i = 0
	g = np.eye(ndim)
	val = np.sum(D*Ms[...,i])

	# Walk along a path along skeleton of Ryskov's polyhedron to minimize M->Tr(DM).
	# Similar to the simplex, but with the skeleton structure pre-processed.
	for niter in range(maxiter):
		vals = np.sum(D[:,:,np.newaxis]*neigh_Ms[i],axis=(0,1))
		ind = vals.argmin() # Find the neighbor of the current vertex with minimal objective
		print(f"Current perfect form : {np.sum(D*Ms[...,i])}, best neighbor : {vals[ind]}")
		if vals[ind]>=val: break # Current vertex is better than all neighbors, hence optimal 
		val = vals[ind]
		data = Ryskov_data[i]
		i = data.neigh_i[ind]
		g = dot(data.neigh_g[...,ind],g)
		D = dot_AtDA(D,data.neigh_g[...,ind].T) # Transform D to account for the change of variables
	else:
		print(f"Voronoi's first reduction failed to converge in {maxiter} iterations")

	Mopt = dot_AtDA(Ms[...,i],g) # Optimal perfect form, normalized so that min vecs have unit norm
	assert np.allclose(val,np.sum(Dold*Mopt))
	return Mopt,(i,g)

def flatten_symmetric_matrix(m):
	"""
	Input : a square (symmetric) matrix.
	Output : a vector containing the lower triangular entries
	"""
	d=m.shape[0]
	assert d==m.shape[1]
	return np.concatenate([m[i,:(i+1)] for i in range(d)],axis=0)

def cross(A):
	"""
	Compute the generalized cross product. 
	A is a matrix of shape (n,n-1,...), representing a family of n-1 vectors.
	"""
	return np.array([(-1)**i*
		np.linalg.det(np.moveaxis(np.delete(A,i,axis=0),(0,1),(-2,-1))) for i in range(len(A))])


def import_prev_run(filename):
	import ast,re
	with open(filename,'r') as s:
		s = s.read()
		for i in range(4): s=s.replace('  ',' ')
		s = s.replace('[ ','[')
		s = s.split("\n")
		status_line = "Linear program found status "
		starts = [i for i,si in enumerate(s) if si.startswith(status_line)]
		s0,sn,sδ = starts[0],len(starts),starts[1]-starts[0]
		status = s[s0:(s0+sn*sδ):sδ]
		status = np.array([int(si[len(status_line)]) for si in status],dtype=int)
		assert np.all((status==0) | (status==2))
		elems = [np.array(ast.literal_eval("".join(s[s0+n*sδ+1:s0+n*sδ+sδ]).replace(' ',','))) for n in range(sn)]
		elems = np.moveaxis(np.array(elems),0,-1)
		print(elems.shape)
		assert status.size==elems.shape[-1]
		return status,elems	

def _np_containssorted(arr,key): 
	"""Wether a numpy sorted array contains a given key"""
	pos = np.searchsorted(arr,key)
	if pos>=arr.size: return False
	return arr[pos]==key

class myset:
	"""
	This class is used to replace the python sets in a memory efficient way for the task at hand.
	(Only really useful for the 5D cases)
	"""
	def __init__(self,init):
		self.external_dtype = init.dtype
		self.external_size = init.shape[0]
		self.uint_t = np.uint64 #np.int16
		dtype_ratio = int(np.iinfo(self.uint_t).bits/np.iinfo(init.dtype).bits)
		self.internal_size = int(np.ceil(self.external_size/dtype_ratio))
		self.internal_dtype = np.dtype([(f"{i}",self.uint_t) for i in range(self.internal_size)])
		self.padding = dtype_ratio*self.internal_size - self.external_size

		if init.ndim==1: self.data = np.empty((0,),dtype=self.internal_dtype)
		else:
			print("Preparing initialization of myset")
			init = np.moveaxis(np.pad(init,((0,self.padding),(0,0))),0,-1).reshape(-1)
			init = np.ascontiguousarray(init)
			print("Sorting")
			self.data = np.sort(init.view(self.internal_dtype))
		self.temp = set()

	def contains(self,key):
		assert np.all(np.abs(key)<=np.iinfo(self.external_size).max)
		key = np.pad(np.reshape(key.astype(self.external_dtype),(self.external_size,)),
			(0,self.padding)).view(self.internal_dtype)
		key_tup = tuple(key.view(self.uint_t))
		return _np_containssorted(self.data,key) or key_tup in self.temp

	def insert(self,keys):
		"""Insert keys, which are not already present."""
		assert np.all(np.abs(keys)<=np.iinfo(self.external_size).max)
		keys = np.moveaxis(np.pad(np.reshape(keys.astype(self.external_dtype),(self.external_size,-1)),
			((0,self.padding),(0,0))),0,-1).view(self.uint_t) #.reshape(-1).view(self.internal_dtype)

		# Insert in the temporary dataset (which is reasonnably small)
		self.temp.update(map(tuple,keys))

		# Merge into the large dataset if temp has become too large
		temp_min_size = 1e7; temp_ratio = 0.05
		if len(self.temp) >= temp_min_size and len(self.temp)>=temp_ratio*len(self.data):
			temp_data = np.sort(np.array(list(self.temp)).view(self.internal_dtype).reshape(-1))
			self.temp = set()
			ii = np.searchsorted(self.data,temp_data)
			self.data = np.insert(self.data,ii,temp_data)


def Perfect1(ndim,iref,retest=False,verbosity=1,init_filename=None,
	linprog_kwargs=None,in_perfect1_tests=None,small_t=np.int16):
	"""
	Compute the set Perfect1(M0), the where M0 is a reference perfect form. It is defined in
	Bonnans, Bonnet, Mirebeau, Monotone discretization of anisotropic differential operators 
	using Voronoi’s first reduction.
	Computing this set allows to bound the size of the vectors appearing in Voronoi's decomposition.
	"""
	if linprog_kwargs is None: linprog_kwargs = dict()
	Ryskov_data = Ryskov_load(ndim)
	# Construct the symmetric matrices DE associated to all subsets of active constraints.
	Ξ = Ryskov_data[iref].Ξ
	E = np.array(list(itertools.combinations(range(Ξ.shape[1]),ndim-1))).T
	E = cross(Ξ[:,E])
	E = E[:,np.any(E,axis=0)]
	D0 = outer_self(E).sum(axis=-1)
	
	Mlcm = np.lcm.reduce([x.M[0,0] for x in Ryskov_data]) # Normalize the minimal value
	Ms = np.stack([x.M*Mlcm/x.M[0,0] for x in Ryskov_data],axis=-1).round().astype(int) 
	GT = np.moveaxis(Ryskov_data[iref].G,0,1) # Perfect1 is invariant under Transposed isometries

	# Optionally, we can tecover the list of all the perfect forms which were discarded 
	# (do not belong to perfect1) in a previous run of the algorithm
	if init_filename is None:
		visited = myset(flatten_symmetric_matrix(D0).astype(small_t))
	else: 
		print(f"Importing output of previous run")
		status,elems = import_prev_run(init_filename)
		print(f"Previous run checked {status.size} equivalence classes"
			"Recovering the {(status==2).size} which were discarded")
		elems = elems[:,:,status==2].astype(small_t)
#		new = list( np.moveaxis(elems,-1,0)[status==0] ) # We would need (i,g) rather than M here
		# When applying the isometries to the previously computed elements, we need to make 
		# We cannot deal with all of them at once, as this would use too much memory (D5 case)
		elems_split = np.array_split(elems,elems.shape[-1]//1000 +1,axis=-1)
		elems_eq = [] # full equivalence classes
		for i,elemsi in enumerate(elems_split):
			print(f"initialization loop {i}/{len(elems_split)}")
			elemsi = dot_AtDA(elemsi[:,:,np.newaxis,:],GT[:,:,:,np.newaxis]).reshape((ndim,ndim,-1))
			elems_eq.append(np.unique(np.moveaxis(flatten_symmetric_matrix(elemsi),0,-1),axis=0))
		elems_eq = np.moveaxis(np.concatenate(elems_eq,axis=0,dtype=small_t),0,-1)
		visited = myset(elems_eq)

	# Testing wether a perfect form belongs to Perfect1(M0) is costly. 
	# We thus tag previously visited forms, as well as their equivalence class under isometries.
	def is_new(M):
		"""Check wether M, or some other isometrically equivalent form, has already been visited"""
		if visited.contains(flatten_symmetric_matrix(M)): return False
		Meq = dot_AtDA(M,GT) # Equivalence class of M modulo the isometry group.
		visited.insert(np.unique(flatten_symmetric_matrix(Meq),axis=-1)) 
		return True

	# Tests wether a perfect form belongs to Perfect1 by solving a linear program
	lps = [] # Save the data associated to previously solved linear programs, just in case
	def in_perfect1(i,g,verb=verbosity):
		M = dot_AtDA(Ms[...,i],g)
		data = Ryskov_data[i]
		neigh_M = dot_AtDA(Ms[...,data.neigh_i],dot(data.neigh_g,g))
		# Constraint Tr(D(μ) M) <= Tr(D(μ) M') for all neighbors M' of M is implemented as Ax<=0
		A = norm2_AV(M,E) - norm2_AV(neigh_M[:,:,:,np.newaxis],E[:,np.newaxis,:])
		b = np.zeros(neigh_M.shape[-1])
		c = np.ones(E.shape[-1]) # Arbitrary objective function
		# We look for a solution x with positive coefficients x>0. By homogeneity, we can look for x>=1
		res = linprog(c,A,b,bounds=(1,None),**linprog_kwargs) 
		# Status : 0 -> feasible, 2 -> unfeasible. 
		# https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html#optimize-linprog-highs
		if res['status'] not in [0,2]: print(f"Issue with linear program : {res['status']=}",
			f"Inputs : {i=}, {g=}. Output : {res}")
		lps.append([(i,g),res['status']]) # (A,b,c),res
		print(f"Linear program found status {res['status']} for \n{M}")
		if verb: print(f"and result {res}")
		return res['status']==0

	# For debug purposes, we can check wether some given perfect forms belong to perfect1
	if in_perfect1_tests is not None:
		if isinstance(in_perfect1_tests,tuple) and len(in_perfect1_tests)==2:
			in_perfect1_tests = [in_perfect1_tests]
		print("Testing data : ",in_perfect1_tests)
		print([in_perfect1(i,g) for (i,g) in in_perfect1_tests])
		return None,None,None

	# Compute Perfect1 iteratively, keeping only a representative of each class.
	_,(i0,g0) = Voronoi(D0); g0 = g0.astype(int)
	assert in_perfect1(i0,g0) 
	is_new(dot_AtDA(Ms[...,i0],g0))
	new = [(i0,g0)]

	perfect1 = []
	while len(new)>0:
		old=new; new = []
		for (i0,g0) in old: 
			# Look at the elements of perfect1 discovered in last step, enumerate their neighbors
			perfect1.append((i0,g0))
			data = Ryskov_data[i0]
			neigh_g = dot(data.neigh_g,g0)
			neigh_M = dot_AtDA(Ms[...,data.neigh_i],neigh_g)
			# Visit all neighbors the neighbors and test if they belong to perfect1
			for (i,g,M) in zip(data.neigh_i,np.moveaxis(neigh_g,-1,0),np.moveaxis(neigh_M,-1,0)):
				if is_new(M) and in_perfect1(i,g): # Note : is_new has a side effect.
					new.append((i,g))

	# For convenience, return the full equivalence classes mod isometries of initial perfect form.
	perfect1_full = [np.unique(dot_AtDA(Ms[...,i],dot(g,GT)),axis=-1) for (i,g) in perfect1]

	if retest: # Optional, was mostly for debug purposes.
		print("--------- Rechecking some of perfect1_full --------")
		for (i,g) in perfect1:
			assert in_perfect1(i,g,verb=0) 
			for gt in np.moveaxis(GT,-1,0)[:20]:
				assert in_perfect1(i,dot(g,gt),verb=0)

	# Basic report
	perfect1_full_n = [x.shape[-1] for x in perfect1_full]
	print(f"Perfect1({Ryskov_data[iref].name}) has {sum(perfect1_full_n)} elements, split in "
		f"equivalence classes of cardinality {perfect1_full_n} modulo isometries of {Ryskov_data[iref].name}")
	return perfect1,perfect1_full,lps


#{'A2':(2,0),'A3':(3,0),'D4':(4,0),'A4':(4,1),'D5':(5,0),'A5':(5,1),'A50':(5,2),'phi0':(6,0),
if __name__ == "__main__":
	if len(sys.argv)<=1:
		print("This program computes the set Perfect1(M0).\n"
		"Inputs : \n "
		"- d=2..5 the dimension. Use at your own risk in dimension d=6. \n"
		"- i=0..I(d) (optional if d<=3) the index of the perfect form.\n" 
		" (Perfect forms ordering : A2, A3, D4,A4, D5,A5,A50, ϕ0,ϕ1,...,ϕ6)\n"
		"- (optional) keyword arguments for scipy.optimize.linprog.\n"
		"Example : python Perfect1.py 4 0 \"{'method':'highs-ds'}\"")
		exit(0)

	d = int(sys.argv[1]) 
	i = int(sys.argv[2]) if d>=4 else 0
	import ast
	linprog_kwargs = None if len(sys.argv)<=3 else ast.literal_eval(sys.argv[3])
	in_perfect1_tests = None if len(sys.argv)<=4 else ast.literal_eval(sys.argv[4])
	init_filename = None if len(sys.argv)<=5 else sys.argv[5]
	
	p1,p1f,lps = Perfect1(d,i,verbosity=0,
		linprog_kwargs=linprog_kwargs,in_perfect1_tests=in_perfect1_tests,init_filename=init_filename)
	print(f"Representatives of Perfect1 modulo isometries : {p1}")
#	print([np.moveaxis(x,-1,0) for x in p1f]) # Show all elements

	exit(0) # Below is only for debug

	# --------- Test Voronoi reduction -------
	ndim=4
	np.random.seed(42)
	v = np.random.normal(size=ndim) 
	v/=np.linalg.norm(v)
	ε = 0.01
	Voronoi(outer_self(v) + ε**2*np.eye(ndim))
