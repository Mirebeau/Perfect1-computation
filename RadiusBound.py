# Copyright 2023 Jean-Marie Mirebeau, Centre Borelli, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This file is devoted to the estimation of the constant from Theorem 4.3 of the paper 
Monotone discretization of anisotropic differential operators using Voronoi’s first reduction
by Bonnans, Bonnet, Mirebeau
(choosing P(M0) = Perfect1(M0), as in the end of S4.1)
"""

import numpy as np
import sys
import Perfect1
from Ryskov import Ryskov_load,norm2_AV,dot_AtDA,dot,outer_self
import ast
import collections


def radius2_bound(ndim,iref,filename=None,use_perfect1_full=False):
	"""
	Compute the radius bound associated to a given perfect form M0, which is defined as 
	R(M0) = sqrt(d) max{|e|_{M1^{-1}} | e in Ξ(M0), M1 in P(M0)}
	"""
	Ryskov_data = Ryskov_load(ndim)
	Mlcm = np.lcm.reduce([x.M[0,0] for x in Ryskov_data]) # Normalize the minimal value
	Ξ = Ryskov_data[iref].Ξ
	Ms = np.stack([x.M*Mlcm/x.M[0,0] for x in Ryskov_data],axis=-1).round().astype(int) 
	GT = np.moveaxis(Ryskov_data[iref].G,0,1) # Perfect1 is invariant under Transposed isometries

	if filename is None or filename == "None": 
		perfect1,perfect1_full,_ = Perfect1.Perfect1(ndim,iref,verbosity=0)
		counts = [x.shape[-1] for x in perfect1_full]
	else: 
		with open(filename,'r') as f:
			s = f.read()
			token = "Representatives of Perfect1 modulo isometries : "
			perfect1 = ast.literal_eval(s[s.find(token)+len(token):].replace('array',''))
			if use_perfect1_full: 
				perfect1_full = [np.unique(dot_AtDA(Ms[...,i],dot(g,GT)),axis=-1) for (i,g) in perfect1]
			start_token,end_token = "split in equivalence classes of cardinality ", " modulo isometries of"
			counts = ast.literal_eval(s[s.find(start_token)+len(start_token):s.find(end_token)])

	if use_perfect1_full:
		M1inv = np.moveaxis(np.linalg.inv(np.moveaxis(np.concatenate(perfect1_full,axis=-1),-1,0))*Mlcm,0,-1)
	else: # We only need a single representative by equivalence class
		M1inv = np.stack([np.linalg.inv(dot_AtDA(Ms[...,i],g))*Mlcm for (i,g) in perfect1],axis=-1)

	radius2_ = np.max(norm2_AV(M1inv[:,:,np.newaxis,:],Ξ[:,:,np.newaxis]))
	print(f"r(M0)^d := max(<e,M1^-1 e>, e in Xi(M0), M1 in Perfect1(M0)) = {radius2_}"
		f" radius bound np.sqrt(d) r(M0) = sqrt({radius2_*ndim}) = {np.sqrt(radius2_*ndim)}")

	# Have a look at some more info
	classes = dict() # Classes by perfect form and cardinality
	classes_i = [list() for _ in Ryskov_data] # Classes by perfect form
	for (i,g),count in zip(perfect1,counts):
		classes_i[i].append((count,g))
		if (i,count) not in classes: classes[(i,count)]=[g]
		else: classes[(i,count)].append(g)
	print(f"Number of equivalence classes by perfect form {[len(x) for x in classes_i]}"
		f" class of : {[data.name for data in Ryskov_data]}")
	print(f"Total number of classes {len(counts)}, and total cardinality of Perfect1 {sum(counts)}")
	print("Number of equivalence classes for each perfect form and cardinality")
	print(sorted([((Ryskov_data[i].name,g),len(val)) for ((i,g),val) in classes.items()]))
	return np.sqrt(radius2_*ndim)

if __name__ == "__main__":
	if len(sys.argv)<=1:
		print("This program prints some information about the set Perfect1(M0).\n"
		"Including the constant appearing in Theorem 4.3, which controls the size of the support"
		" of Voronoi's decomposition of a matrix.\n"
		"Inputs : \n "
		"- d=2..5 the dimension. Use at your own risk in dimension d=6. \n"
		"- i=0..I(d) (optional if d<=3) the index of the perfect form.\n" 
		" (Perfect forms ordering : A2, A3, D4,A4, D5,A5,A50, ϕ0,ϕ1,...,ϕ6)\n"
		"- (optional) The name of a file containing the terminal output of Perfect1.py d i.\n")
		exit(0)

	ndim = int(sys.argv[1])
	iref = 0 if len(sys.argv)<=2 and ndim<=3 else int(sys.argv[2])
	filename = None if len(sys.argv)<=3 else sys.argv[3]
	use_perfect1_full = False if len(sys.argv)<=4 else ast.literal_eval(sys.argv[4])
	radius2_bound(ndim,iref,filename,use_perfect1_full)
