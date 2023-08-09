## Compute Jacobian_mpm


Given n grid velocites v_i, and a particular particle p, we want to find J such that v_p = J * v_i.
All velocites are in world frame.
Here v_p has dimension 3by1, J has dimension 3by(3n), v_i has dimension (3n)by1

implementation:

given particle x_p^0
sort particles and grid nodes, compute weights(x_p^0)

for each of the 27 neighbor grid node j, form a 3by3 matrix 
[ w_jp 0     0]
[ 0   w_jp   0]
[ 0   0   w_jp]

Insert this matrix into corresponding spot, using grid node j's global index.

TBD: cache in CalcMpmContact. Schur complement??
