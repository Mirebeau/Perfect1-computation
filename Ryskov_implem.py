# Copyright 2023 Jean-Marie Mirebeau, Centre Borelli, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This files generates the structure of Ryskov's polyhedron in dimension d<=6 :
all support vectors, isometry groups, and neighbors. 
We do not start from scratch, but from the known expression of each perfect form and a representative
of each class of neighbors modulo isometries of the perfect form. 
Based on the papers :
 - Conway, J. H. & Sloane, N. J. A. Low-Dimensional Lattices. III. Perfect Forms. Proceedings of the Royal Society of London A: Mathematical, Physical and Engineering Sciences 418, 43–80 (1988).
 - Complete enumeration of the extreme senary forms*, Barnes, 1957
"""

import sys; sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")
import numpy as np
from numpy import expand_dims as ed
from collections import namedtuple
import itertools
from Ryskov import Ryskov_data,dot,norm2_AV,dot_AtDA,outer_self

def matrix_perm(perm):
	"""Generates a permutation matrix from a permutation"""
	dim=len(perm)
	m = np.zeros((dim,dim))
	for i,j in enumerate(perm):
		m[i,j]=1
	return m

np.set_printoptions(linewidth=1000)

# This data is known from the literature, see the two cited papers.
Ryskov_base = namedtuple('Ryskov',(
	'name', # A customary name, see Conway, Sloane, Low-dimensional Lattices
	'M', # The perfect form. The vector (1,0,...,0) must be minimal
	'neigh', # A list of representatives of each equivalence class of neighbors modulo isometries, 
			 # given as a pairs (i,g) where i is the index of the perfect form, and g in GLdZ
	))

# The dimension 1 case is quite degenerate (a single perfect form, without any neighbors)
Ryskov_base1 = [
Ryskov_base(
	'A1', ((1,),),
	[ ]
	)
]

Ryskov_base2 = [
Ryskov_base( 
	'A2', ((2,1),(1,2)), 
	[ (0,((1,0),(1,1))) ]
	)
]

Ryskov_base3 = [
Ryskov_base( # Perfect form known as A3
	'A3', ((2,1,1),(1,2,1),(1,1,2)),
	[ (0,((-1,0,0),(1,1,0),(1,0,1)) ) ]
	)
]

Ryskov_base4 = [
Ryskov_base( # Perfect form known as D4
	'D4', ((2,1,1,0),(1,2,1,1),(1,1,2,1),(0,1,1,2)),
	[( 1,((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)) ),
	 ( 0,((-1,0,0,0),(0,-1,0,0),(1,0,1,0),(0,1,0,1)) ) ]
	 ),
Ryskov_base( # Perfect form known as A4
	'A4', ((2,1,1,1),(1,2,1,1),(1,1,2,1),(1,1,1,2)),
	[ (0,((1,0,0,0),(0,1,0,0),(-1,0,1,0),(1,0,0,1))) ]
	)
]

Ryskov_base5 = [
Ryskov_base(
	'D5',((2,1,1,1,0),(1,2,1,1,1),(1,1,2,1,1),(1,1,1,2,1),(0,1,1,1,2)),
	[ (0, ((1,1,0,0,0),(-1,0,0,0,0),(0,-1,1,0,0),(0,0,0,1,0),(1,1,0,0,1)) ),
	(0,((-1,0,0,0,0),(1,1,1,0,0),(0,-1,0,-1,-1),(0,0,-1,0,0),(0,0,0,1,0))),
	(2, ((0,-1,0,0,0),(1,0,1,0,0),(1,0,0,1,0),(1,0,0,0,0),(1,0,1,1,1)) ),
	(1, ((1,0,0,0,0),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(0,0,0,0,1)) )	]
	),
Ryskov_base( 
	'A5', ((2,1,1,1,1),(1,2,1,1,1),(1,1,2,1,1),(1,1,1,2,1),(1,1,1,1,2)),
	[ (0, ((1,1,0,0,0),(-1,0,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(1,1,0,0,1)) ) ]
	),
Ryskov_base( # Note that the minimal vectors squared norm is 4
	'A50', ((4,1,1,-2,-2),(1,4,1,-2,-2),(1,1,4,-2,-2),(-2,-2,-2,4,1),(-2,-2,-2,1,4)),
	[ (0, ((0,0,0,-1,0),(1,0,0,0,0),(0,1,0,0,0),(0,0,1,0,0),(-1,-1,-1,0,1)) ) ]
	),
]

# -----------


A12 = ((1,1,0,0,1,1), (1,0,0,0,0,0), (0,1,0,0,0,0), (0,0,1,0,0,0), (0,0,0,1,0,0), (-1,-1,-1,0,0,-1))
embed1 = ((1,1,1,1,1,1), (1,-1,0,0,0,0), (0,0,1,0,0,0), (0,0,0,1,0,0), (0,0,0,0,1,0), (0,0,0,0,0,1.))

Ryskov_base6 = [
Ryskov_base(
	'ϕ0', 
((2, 1, 1, 1, 1, 1),
 (1, 2, 1, 1, 1, 1),
 (1, 1, 2, 1, 1, 1),
 (1, 1, 1, 2, 1, 1),
 (1, 1, 1, 1, 2, 1),
 (1, 1, 1, 1, 1, 2)),
[ (1,np.eye(6)) ]
),
Ryskov_base(
	'ϕ1',
((2, 0, 1, 1, 1, 1),
 (0, 2, 1, 1, 1, 1),
 (1, 1, 2, 1, 1, 1),
 (1, 1, 1, 2, 1, 1),
 (1, 1, 1, 1, 2, 1),
 (1, 1, 1, 1, 1, 2)),
[ (0,np.eye(6)),
(1,matrix_perm((2,3,0,1,4,5))),
(2,np.eye(6)),
(2,matrix_perm((2,3,4,0,1,5))),
(2,A12),
(3,np.eye(6)),
(4,np.eye(6)),
(5,np.eye(6)) ]
),
Ryskov_base(
	'ϕ2',
((2, 0, 0, 1, 1, 1),
 (0, 2, 1, 1, 1, 1),
 (0, 1, 2, 1, 1, 1),
 (1, 1, 1, 2, 1, 1),
 (1, 1, 1, 1, 2, 1),
 (1, 1, 1, 1, 1, 2)),
[ (2,dot(
	((1,1,0,0,1,1),
	 (1,0,0,0,0,0),
	 (0,1,0,0,0,0),
	 (0,0,1,0,0,0),
	 (0,0,0,1,0,0),
	 (-1,-1,-1,0,0,-1)),
	   matrix_perm( (4,5,0,1,2,3) ) ) ), 
(2,((1,0,0,0,0,0),
	(1,-1,0,0,0,0),
	(0,0,1,0,0,0),
	(0,0,0,1,0,0),
	(0,0,0,0,1,0),
	(-1,1,0,0,0,1) ) ),
(2,((1,0,0,0,0,0),
	(2,1,0,1,1,1),
	(1,0,1,1,0,1),
	(-1,0,0,-1,0,0),
	(0,0,0,0,1,0),
	(-2,0,0,0,-1,-1)) ) ]
),
Ryskov_base(
	'ϕ3',
((4, 1, 2, 2, 2, 2),
 (1, 4, 2, 2, 2, 2),
 (2, 2, 4, 1, 2, 2),
 (2, 2, 1, 4, 2, 2),
 (2, 2, 2, 2, 4, 1),
 (2, 2, 2, 2, 1, 4)),
[ (1,matrix_perm((4,5,0,1,2,3))),
(2,dot(A12, matrix_perm((4,5,0,2,1,3)) ) ),
(5,((1,0,1,0,0,0),
	(1,0,0,1,0,0),
	(-1,0,0,0,0,0),
	(0,0,0,0,1,0),
	(0,0,0,0,0,1),
	(0,1,0,0,0,0)) ) ]
),
Ryskov_base(
	'ϕ4',
((4, 1, 2, 2, 2, 2),
 (1, 4, 2, 2, 2, 2),
 (2, 2, 4, 1, 1, 1),
 (2, 2, 1, 4, 1, 1),
 (2, 2, 1, 1, 4, 1),
 (2, 2, 1, 1, 1, 4)),
[(1,np.eye(6)),
(2,((1,0,0,0,0,0),
	(0,1,0,0,0,0),
	(-1,-1,-1,-1,-1,-1),
	(0,0,0,1,0,0),
	(0,0,0,0,1,0),
	(0,0,0,0,0,1.)) ),
(5, dot(np.linalg.inv(embed1),dot(
   ((0,0,0,0,1,0),
	(0,1,0,0,0,0),
	(0,0,1,0,0,0),
	(0,0,0,1,0,0),
	(1,0,0,0,0,0),
	(0,0,0,0,0,-1)),
	embed1 )) )
]
),
Ryskov_base(
	'ϕ5',
((4, 1, 2, 2, 2, 2),
 (1, 4, 2, 2, 2, 2),
 (2, 2, 4, 1, 1, 2),
 (2, 2, 1, 4, 1, 2),
 (2, 2, 1, 1, 4, 2),
 (2, 2, 2, 2, 2, 4)),
[(1,np.eye(6)),
(2,((1,0,0,0,0,0),
	(0,1,0,0,0,0),
	(-1,-1,-1,-1,-1,-1),
	(0,0,0,1,0,0),
	(0,0,0,0,1,0),
	(0,0,0,0,0,1.)) ),
(2,matrix_perm( (2,3,4,0,1,5) )),
(3,dot(dot(dot(np.linalg.inv(embed1),
	((0,0,0,1,0,0),
	 (0,1,0,0,0,0),
	 (0,0,1,0,0,0),
	 (1,0,0,0,0,0),
	 (0,0,0,0,-1,0),
	 (0,0,0,0,0,-1.)) ),embed1),
	matrix_perm((0,1,4,5,2,3)) 
)),
(4,((1,0,0,0,0,1),
	(0,1,0,0,0,1),
	(0,0,1,0,0,0),
	(0,0,0,1,0,0),
	(0,0,0,0,1,0),
	(0,0,0,0,0,-1.)) ),
]
),
Ryskov_base(
	'ϕ6',
((4, 0, 1, 2, 2, 1),
 (0, 4, 2, 2, 1, 2),
 (1, 2, 4, 2, 2, 2),
 (2, 2, 2, 4, 2, 1),
 (2, 1, 2, 2, 4, 0),
 (1, 2, 2, 1, 0, 4)),
[(2,np.eye(6))]
)
]

# Add the missing neighbors of ϕ2 (using symmetry of the neighbor relation)
for neigh_i,base in enumerate(Ryskov_base6):
	if base.name=='ϕ2': continue
	for neigh_j,neigh_g in base.neigh:
		if neigh_j!=2: continue
		Ryskov_base6[2].neigh.append((neigh_i,np.linalg.inv(neigh_g).round()))

Ryskov_base_ = [None,Ryskov_base1,Ryskov_base2,Ryskov_base3,
			Ryskov_base4,Ryskov_base5,Ryskov_base6]

# ------------- Computation of the full perfect form data -----------

def minimal_vectors(M):
	"Compute the minimal_vectors of a perfect form M."

	# Build a list of candidate minimal vectors. Their norm must be below that of (1,0,...,0)
	R = np.sqrt(M[0,0] * np.linalg.norm(np.linalg.inv(M)))
	R = int(np.ceil(R+1e-6))
	ndim = len(M)
	candidates = list(itertools.product(range(R),*(range(-R,R),)*(ndim-1) ))
	candidates = np.array(candidates,dtype=float).T

	# Select the candidates whose norm is minimal
	norm2s = norm2_AV(M,candidates)
	select = norm2s == np.min(norm2s[norm2s!=0]) # The good ones
	Ξ = candidates[:,select]

	# Remove candidates which are below 0 lexicographically
	pos = (Ξ!=0).argmax(axis=0) # index of first non-zero entry
	select = np.take_along_axis(Ξ,pos[np.newaxis],0)[0] > 0
	Ξ = Ξ[:,select]

	# Sort in a convenient way
	Ξ_ = np.concatenate((-Ξ[::-1],[np.sum(np.abs(Ξ),axis=0)]),axis=0)
	ind = np.lexsort(Ξ_)
	Ξ = Ξ[:,ind]

	return Ξ

def isometry_group(M,Ξ):
	"""Compute the isometry group of a reference perfect form M, with minimal vectors Ξ."""
	ndim = len(M)
	# For simplicity, assume the canonical basis are minimal vectors 
	# This is the case of all the reference perfect forms tabulated above 
	assert np.allclose(Ξ[:,:ndim],np.eye(ndim)) 

	# An isometry must send any basis of minimal vectors to 
	# another basis of minimal vectors, possibly permuted and with sign changes
	Gi = np.array(list(itertools.combinations(range(Ξ.shape[1]),ndim))).T
	G = np.take_along_axis(ed(Ξ,axis=-1),ed(Gi,axis=0),axis=1).astype(int)
	det = np.linalg.det(np.moveaxis(G,-1,0)).round() # The determinant must be integer
	G = G[...,np.abs(det)==1]

	# We have not yet accounted for permutations and sign changes, but we can already 
	# test wether the set of absolute values of the entries are correct 
	M_chg = dot_AtDA(M,G)
	select = np.all(np.sort(np.abs(M_chg).reshape((ndim**2,-1)),axis=0) 
		== np.sort(np.abs(M).reshape(-1))[:,np.newaxis],axis=0)
	G = G[...,select]

	# We now check wether one can attribute signs so that the set of original entries is recovered
	signs = np.array(list(itertools.product((-1,1),repeat=ndim))).T
	msigns = outer_self(signs)
	G = (G[...,np.newaxis]*signs[:,np.newaxis,:]).reshape((ndim,ndim,-1))
	M_chg = dot_AtDA(M,G)
	select = np.all(np.sort(M_chg.reshape(ndim**2,-1),axis=0)
		== np.sort(M.reshape(-1))[:,np.newaxis],axis=0)
	G = G[...,select]

	# We now must find the correct permutation 
	perm = np.array(list(itertools.permutations(range(ndim)))).T
	G = G[:,perm].reshape((ndim,ndim,-1))
	M_chg = dot_AtDA(M,G)
	select = np.all(M_chg==M[...,np.newaxis],axis=(0,1))
	G = G[...,select]

	# Eliminate duplicate elements, if any
	G = np.unique(G.reshape((ndim**2,-1)),axis=1).reshape((ndim,ndim,-1))
	return G


def Ryskov_complete(Ryskov_base):
	"""Complete the information regarding the perfect forms in a given dimension."""
	full = [] # The complete information
	for base in Ryskov_base:
#		if base.name not in ["ϕ0"]: continue
		M = np.array(base.M,dtype=float)
		ndim = len(M)
		symdim = (ndim*(ndim-1))//2

		# Compute the minimal vectors
		Ξ = minimal_vectors(M)
		print(f"Found {Ξ.shape[1]} minimal vectors for perfect form {base.name}.")
		print(Ξ)

		# Check the base neighbors. They must share enough minimal vectors.
		for i,(neigh_i,neigh_g) in enumerate(base.neigh):
			neigh_M = np.array(Ryskov_base[neigh_i].M,dtype=float)
			N = dot_AtDA(neigh_M,neigh_g)
			norm2s = norm2_AV(N,Ξ) 
			sharedmin = np.sum(norm2s==neigh_M[0,0])
			print(f"Base neighbor {i} shares {sharedmin} minimal vectors")
			assert sharedmin>=symdim-1

		# Compute the unimodular isometry group
		G = isometry_group(M,Ξ)
		print(f"Found unimodular isometry group with {G.shape[-1]} elements")

		# Compute all the neighbors, by applying the isometries to the set of representatives
		neigh = []
		for i,(neigh_i,neigh_g) in enumerate(base.neigh):
			neigh_M = np.array(Ryskov_base[neigh_i].M,dtype=float)
			N = dot_AtDA(neigh_M,neigh_g)
			N = dot_AtDA(N,G) # All the neighbors, with repetitions, removed below
			_,ind = np.unique(N.reshape(ndim**2,-1),axis=1,return_index=True)
			Gprod=dot(neigh_g,G[:,:,ind])
			neigh.append((neigh_i,Gprod)) 
			print(f"Found {len(ind)} neighbors of class {i}")
		neigh_i = np.concatenate( [np.full(g.shape[-1],i) for i,g in neigh])
		neigh_g = np.concatenate( [g for i,g in neigh],axis=-1)
		print(f"Total number of neighbors",neigh_i.size)
		assert np.max(np.abs(neigh_g))<128 and np.max(np.abs(G))<128 # Fits in int8
		full.append(Ryskov_data(base.name,*[e.astype(int) for e in (M,Ξ,neigh_i,neigh_g,G)]))
		print("")

	return full