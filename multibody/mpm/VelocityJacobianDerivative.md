## Compute the Jacobian that transforms all dofs to contact velocity

Say we have already identified a particle 
$$x_{pC}, v_{pC}$$
that penetrates a rigid body A.

Let's first compute the Jacobian J s.t.
$$ v_{pC} = J \cdot v_i$$
where we have n grid nodes.

J will have dimension 3 by 3n. (Confirm that?)

Since the computation of ``AppendContactKinematics`` is called after we have computed the free motion state, we can assume that we have already had sorted grid nodes and sorted particles, the weights 
$$w_{ip}$$
have also been computed.

Nevertheless, since we only have access to ``Context`` and we only have ``MpmState`` which is just ``Particles``, those info are lost.

So here we need to redo the grid and setup transfer thing (to sort the grid nodes and get weights) -- possible issue here tbd

Anyways let's say we have the transfer setup now.

To compute the J, we loop over all 27 neighboring grid nodes, denoted its index among all grid nodes to be j. The 3by3 block ``J.block<3,3>(0, 3*j)`` is a diagonal matrix with wjp on the diagonal.