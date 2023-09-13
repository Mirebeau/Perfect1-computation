# In this file, following the suggestion of a referee, we have a look at Voronoi's decomposition of 
# (the inverses of) perfect forms. We recover that some are eutactic, but apart from this the 
# experiment is not very conclusive. 

# This file will not execute correctly on a computer other than mine (contrary to the rest of the
# repo) : the paths must be changed.

import sys 
sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations")

from Ryskov_implem import Ryskov_base_
from Perfect1 import Voronoi
from agd.Eikonal import VoronoiDecomposition
import numpy as np
import agd

agd.Eikonal.LibraryCall.binary_dir['FileVDQ']="/Users/jean-mariemirebeau/bin/FileVDQ/Release"

for (d,data) in enumerate(Ryskov_base_): # Dimension of space
	if d<2: continue
#	if d>4: continue
	print(f"---------- Dimension {d=} ----------")
	for (i,form) in enumerate(data):
		print(f"----------- {form.name} --------------")
		print(form.M)

		Minv = np.linalg.inv(form.M)
		Mopt,(iopt,g) = Voronoi(Minv)

		print(f"Opt for M^-1 is M : {np.allclose(np.array(form.M)/form.M[0][0], Mopt)} ") 
		chg,vertex,objective,weights,offsets = VoronoiDecomposition(Minv,steps="Split")
		true_offsets = np.any(offsets,axis=0)
		offsets = offsets[:,true_offsets]
		weights = weights[true_offsets]
		print(weights,objective/2)
		print(f"Minimal weight (positive implies eutactic) : {np.min(weights)}")
