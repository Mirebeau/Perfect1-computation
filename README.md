## Companion code for the manuscript
## Bonnans, Bonnet, Mirebeau, Monotone discretization of anisotropic differential operators using Voronoi’s first reduction

This code corresponds to S 4.1 of the paper.

Features : 
- Computes the support vectors, isometry groups, and neighbor relations on Ryskov's polyhedron, for all perfect forms in dimension d<=6. (Using known tabulated data as a starting point.)
- Computes Voronoi's reduction of a symmetric positive definite matrix, in dimension d<=6.
- Computes the set Perfect1 associated to all perfect forms in dimension d<=4. (The code will not terminate in dimension d=5,6 in a reasonnable time.)

Notes : 
- This code does NOT implement the matrix decomposition associated with Voronoi's reduction. See the [AGD library](https://github.com/Mirebeau/AdaptiveGridDiscretizations) instead for that purpose.

Usage : 
- Make sure you have python with scipy installed.
- `python Ryskov.py` and `python Perfect1.py` will show the input argument syntax.
