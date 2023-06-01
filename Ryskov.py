# Copyright 2023 Jean-Marie Mirebeau, Centre Borelli, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This files loads the structure of Ryskov's polyhedron (precomputed or recomputed)
"""
import sys
from collections import namedtuple
import numpy as np

# This data is reconstructed.
Ryskov_data = namedtuple('Ryskov',(
	'name', # A customary name
	'M', # The perfect form
	'Ξ', # The minimal vectors
	'neigh_i', # The class of all neighbors
	'neigh_g', # The changes of variables associated to all neighbors
	'G', # The group of unimodular isometries of the perfect form
	))

# -----------
# We use a "geomety first" convention : a collection of n matrices of shape (n,n)
# is represented as a ndarray of shape (d,d,n). This has both advantages and disadvantages. 
# The following functions work with this convention.
# ----------

def common_field(arrays,depths):
	"""Adds trailing dimensions consistently to the arrays."""
	arrays = [np.asarray(arr) for arr in arrays]
	n = max(arr.ndim-depth for (arr,depth) in zip(arrays,depths))
	return [arr.reshape(arr.shape+(1,)*(n-arr.ndim+depth)) for (arr,depth) in zip(arrays,depths)]

def dot(A,B):
	"""Matrix product"""
	A,B = common_field((A,B), depths=(2,2))
	return np.sum(A[:,:,np.newaxis]*B[np.newaxis,:,:],axis=1)
def norm2_AV(A,V):
	"""V^t A V"""
	A,V = common_field((A,V), depths=(2,1))
	return np.sum(V[:,np.newaxis]*A*V[np.newaxis,:],axis=(0,1))
def dot_AtDA(D,A):
	"""A^t D A. Computes the quadratic form congruent to D by the change of variables A"""
	D,A = common_field((D,A), depths=(2,2))
	return dot(np.moveaxis(A,0,1),dot(D,A))
def outer_self(V):
	"""V V^t"""
	return V[:,np.newaxis]*V[np.newaxis,:]

# -------- Save and reload data --------

def tolist(data):
	if isinstance(data,list): return [tolist(x) for x in data]
	elif isinstance(data,tuple): return tuple(tolist(x) for x in data)
	elif isinstance(data,np.ndarray): return data.tolist()
	elif isinstance(data,dict): return tolist(list(data.items()))
	elif isinstance(data, np.integer):return int(data)
	elif isinstance(data,Ryskov_data): return [tolist(x) for x in data]
	else: return data


def Ryskov_load(d,precomputed=True,save=True):
	"""
	Tries to load the data describing the Ryskov polyhedron. 
	Will be recomputed, and saved, if not present. 
	- d : dimension of the space
	"""
	import json
	if precomputed: # Try to load saved data
		try:
			with open(f"Ryskov{d}.dat",encoding='utf8') as f:
				data = json.load(f)
			return [Ryskov_data(x[0],*map(np.array,x[1:])) for x in data]
		except IOError: 
			print("(Re)computing Ryskov polyhedron structure data, which could not be loaded")
 
	from Ryskov_implem import Ryskov_complete,Ryskov_base_
	data = Ryskov_complete(Ryskov_base_[d])
	if save:
		try:
			with open(f"Ryskov{d}.dat",'w',encoding='utf8') as f:
				json.dump(tolist(data),f,ensure_ascii=False,separators=(',', ':'))
		except IOError: print("Could not save Ryskov polyhedron structure data")

	return data


if __name__ == "__main__":
	if len(sys.argv)<=1:
		print("This program computes and saves the structure of Ryskov's polyhedron.\n"
		"Input : d=2..6 the desired dimension. Use d=0 for all dimensions")
		exit(0)
	if len(sys.argv)==2:
		ndim = int(sys.argv[1])
		if 2<=ndim<=6: Ryskov_load(ndim,precomputed=False,save=True)
		elif ndim==0: 
			for d in range(2,7):
				Ryskov_load(d,precomputed=False,save=True)
		else: raise InputError("Unsupported dimension")
