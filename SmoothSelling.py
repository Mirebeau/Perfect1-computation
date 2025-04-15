"""
Companion code for the manuscript : 
"Anisotropic Waves and Selling's decomposition"

Given a perfect form M_0, this codes compute the a finite set P_0(M_0), and a constant c(M_0), such that
Tr(D M)^d >= Tr(D M_0)^d + c(M_0) det(D), for all M notin P_0(M_0), and all D for which M_0 is optimal.

It does *not* compute the smooth selling decomposition, which is implemented in the AGD library.
"""

#import sys 
#sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")
#import agd

import numpy as np
import itertools
from Ryskov import Ryskov_load,norm2_AV,dot_AtDA,dot,outer_self
from Perfect1 import flatten_symmetric_matrix,myset

def ComputeP0(ndim=2,iref=0,useMyset=False,small_t=np.int16):
	"""
	- ndim : dimension
	- irefl : index of the perfect form of interest
	"""
	Ryskov_data = Ryskov_load(ndim)
	# Ms : the list of perfect forms, normalized so that minimal vectors have norm one
	Mlcm = np.lcm.reduce([x.M[0,0] for x in Ryskov_data]) # Normalize the minimal value
	Ms = np.stack([x.M*Mlcm/x.M[0,0] for x in Ryskov_data],axis=-1).round().astype(int) 
	G = Ryskov_data[iref].G 
	Ξ = Ryskov_data[iref].Ξ.astype(int)


	print("Perfect form of interest : ",Ms[:,:,iref])
	print("Minimal vectors ", Ξ)
	assert np.allclose(norm2_AV(Ms[:,:,iref],Ξ),Mlcm) # Minimal vectors for the perfect form

	# By the Cauchy-Binet formula, one has
	# det( sum_{e in Ξ} λ(e) e e^T ) = sum_{ {e_1,...,e_d} subset Ξ } λ(e_1) ... λ(e_d) det(e_1,...,e_d)^2
	# We identify below the monomials appearing in this sum.
	nΞ = Ξ.shape[1]
	det_terms = np.array(list(itertools.combinations(range(nΞ),ndim)))
	det_values = np.array([np.linalg.det(Ξ[:,term]) for term in det_terms])
	pos = det_values!=0
	det_values = det_values[pos]
	det_terms = det_terms[pos]

	print("Monomials in the Cauchy-Binet formula :\n", det_terms)
	print("Coefficients (to be squared) : ", det_values)

	def c(M):
		"""
		Compute the minimum of (<e_1,M e_1> ... <e_d,M e_d> - 1) / det(e_1,...,e_d)^2,
		among all terms identified above.
		"""
		eMe = norm2_AV(M,Ξ)/Mlcm		
#		print(eMe, eMe[det_terms])
#		print(np.prod(eMe[det_terms],axis=1) )
		return np.min( (np.prod(eMe[det_terms],axis=1) - 1)/det_values**2 )

	if useMyset: # Memory efficient implementation
		visited = myset(flatten_symmetric_matrix(Ms[...,iref]).astype(small_t))
		def is_new(M): # This version is for 'myset', handling large datasets
	 		if visited.contains(flatten_symmetric_matrix(M)): return False
	 		Meq = dot_AtDA(M,G) # Equivalence class of M modulo the isometry group.
	 		visited.insert(np.unique(flatten_symmetric_matrix(Meq),axis=-1)) 
	 		return True
	else: # Standard python sets
		visited = set()
		def is_new(M):
			if tuple(flatten_symmetric_matrix(M)) in visited: return False
			Meq = dot_AtDA(M,G) # Equivalence class of M modulo the isometry group.
			for Meq_ in np.moveaxis(Meq,-1,0): visited.add(tuple(flatten_symmetric_matrix(Meq_)))
			return True

	# Explore Ryskov's skeleton graph, starting from M0, stopping at perfect forms for which c(M)!=0. 
	assert is_new(Ms[...,iref])
	new = [(iref,np.eye(ndim).astype(int))]
	P0 = [] # The set P0(M0) that we are computing
	c0 = np.inf # The associated optimal constant
	while len(new)>0:
		old=new; new = []
		for (i0,g0) in old:
			P0.append((i0,g0))
			data = Ryskov_data[i0]
			neigh_g = dot(data.neigh_g,g0) # Changes of variables to the neighbors
			neigh_M = dot_AtDA(Ms[...,data.neigh_i],neigh_g) # Neighbor perfect forms
			# Visit all the neighbors, and test if they below to P0
			for (i,g,M) in zip(data.neigh_i,np.moveaxis(neigh_g,-1,0),np.moveaxis(neigh_M,-1,0)):
				if is_new(M): # Note : is_new has a side effect.
#					print(M)
					c_M = c(M) 
					assert c_M>=0
					if c_M>0: c0 = min(c0,c_M)
					else: new.append((i,g))

	# return the full set P0, taking into account all isometries of the inital perfect form
	P0_full = [np.unique(dot_AtDA(Ms[...,i],dot(g,G)),axis=-1) for (i,g) in P0]

	# Basic report
	P0_full_n = [x.shape[-1] for x in P0_full]
	report = f"P0({Ryskov_data[iref].name}) has {sum(P0_full_n)} elements, split in {len(P0)} " \
		f"equivalence classes of cardinality {P0_full_n} modulo isometries of {Ryskov_data[iref].name}\n" \
		f"Optimal constant is {c0=}, showing one representative of each class"
#	print(f"P0({Ryskov_data[iref].name}) has {sum(P0_full_n)} elements, split in {len(P0)} "
#		f"equivalence classes of cardinality {P0_full_n} modulo isometries of {Ryskov_data[iref].name}")
#	print(f"Optimal constant is {c0=}, showing one representative of each class")
	print(report)
	print(np.array([dot_AtDA(Ms[...,i],g) for (i,g) in P0]))
	return report, c0, P0, P0_full

def to_list(data):
	if isinstance(data,np.ndarray): return data.astype(int).tolist()
	elif isinstance(data,list): return [to_list(x) for x in data]
	elif isinstance(data,tuple): return tuple([to_list(x) for x in data])
	elif isinstance(data,np.int64): return int(data)
	else: return data 

def json_export(data,filename):
	import json
	with open(f'Results_SmoothSelling/{filename}.json', 'w', encoding='utf-8') as f:
		json.dump(to_list(data), f, ensure_ascii=False, indent=4)


# Uncomment any of the following lines to compute P_0(M_0), and find c_0(M_0)

#json_export(ComputeP0(2),"P0_2") # 4 Elements, c0=2
#json_export(ComputeP0(3),"P0_3") # 127 elements, c0=1
#json_export(ComputeP0(4,0,useMyset=True),"P0_40") # 31409 elements, 86 eq classes, c0=1
#json_export(ComputeP0(4,1,useMyset=True),"P0_41") # 14848 elements, 156 eq classes, c0=1
#json_export(ComputeP0(5,0,useMyset=True),"P0_50") # P0(D5) has 31006233 elements, split in 17427 equivalence classes of cardinality [1, 240, 80, 40, 40, .... 960, 480, 480] modulo isometries of D5. Optimal constant is c0=0.25
#json_export(ComputeP0(5,1,useMyset=True),"P0_51") # P0(A5) has 6713695 elements, split in 9991 equivalence classes of cardinality [1, 15, 180, ... 360, 360, 360, 180, 60, 120, 720, 360, 720, 360, 360] modulo isometries of A5. Optimal constant is c0=0.5
#json_export(ComputeP0(5,2,useMyset=True),"P0_52") # P0(A50) has 8910763 elements, split in 13519 equivalence classes of cardinality [1, 15, 180, 180, 180,..... 180, 60] modulo isometries of A50. Optimal constant is c0=0.3125

# We check below that the support Ξ(M_0) contains the canonical basis for each reference perfect form
for ndim in [2,3,4,5,6]:
	Ryskov_data = Ryskov_load(ndim)
	for iref,data in enumerate(Ryskov_data):
		print(f"Perfect for {data.name}=\n{data.M}\nhas support Ξ=\n{data.Ξ}")
		assert(np.allclose(data.Ξ[:,:ndim],np.eye(ndim)))
		print("which starts with the canonical basis")