## Companion code for the manuscript
## Bonnans, Bonnet, Mirebeau, Monotone discretization of anisotropic differential operators using Voronoi’s first reduction

This code corresponds to S 4.1 of the paper.

Features : 
- Computes the support vectors, isometry groups, and neighbor relations on Ryskov's polyhedron, for all perfect forms in dimension d<=6. (Using known tabulated data as a starting point.)
- Computes Voronoi's reduction of a symmetric positive definite matrix, in dimension d<=6.
- Computes the set Perfect1 associated to all perfect forms in dimension d<=5. (The code is instantaneous in dimension <=4, but takes a day of computation d=5 for the perfect form D5, and a few minutes in the other cases. It does not terminate in reasonnable time in dimension d=6.)

Notes : 
- This code does NOT implement the matrix decomposition associated with Voronoi's reduction. See the [AGD library](https://github.com/Mirebeau/AdaptiveGridDiscretizations) instead for that purpose.

Usage : 
- Make sure you have python with scipy installed.
- Run `python Ryskov.py`, `python Perfect1.py` and `python RadiusBound.py` in the command line.
This will show the input argument syntax.
