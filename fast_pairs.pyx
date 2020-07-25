from cython import wraparound, boundscheck
from libc.math cimport sqrt, cos, sin

import numpy as np
# cimport numpy as np


@boundscheck(False)
@wraparound(False)
def cy_nhat(double[:] alpha, double[:] delta, double[:,:] nhat):
	cdef int N = alpha.size
	cdef int i
	for i in range(N):
		nhat[i,0] = cos(delta[i]) * cos(alpha[i])
		nhat[i,1] = cos(delta[i]) * sin(alpha[i])
		nhat[i,2] = sin(delta[i])
	return np.asarray(nhat)


@boundscheck(False)
@wraparound(False)
def cy_pairs(double[:] chi, double[:,:] nhat, \
			double[:,:] r_arr, double[:,:] cth1_cth2_arr, double[:,:] sth1_sth2_arr):

	"""
	This function computes for all pairs of N ordered objects
	the physical separation and the two angles theta1 and
	theta2 that give the angular separation of the two lines
	of sight wtih respect to the connecting vector x2 - x1
	(see figure 9.8 in Dodelson cosmology book).

	There are N(N-1)/2 unique pairs and we represent these
	on a matrix so that the pair consisting of the ith and
	jth object are assigned to the (i,j) and (j,i) element.
	The diagonal corresponds to self-pairs; these are
	assigned a constant value.

	Parameters
	-------------------------------------------------------
	chi:			(N,) array of comoving distances
	nhat:			(N,3) array of unit LOS vectors
	r_arr:			(N,N) array (zero matrix for init)
	cth1_cth2_arr:	(N,N) array (zero matrix for init)
	cth1_cth2_arr:	(N,N) array (zero matrix for init)

	Returns
	-------------------------------------------------------
	r_arr:			(N,N) array
	cth1_cth2_arr:	(N,N) array
	sth1_sth2_arr:	(N,N) array

	The (i,j) and (j,i) components of cth1_cth2_arr is

		cos(theta_1[i,j]) * cos(theta_2[i,j])

	and similarly for sth1_sth2_arr

		sin(theta_1[i,j]) * sin(theta_2[i,j])

	"""
	cdef int i,j
	cdef int N = chi.size
	cdef double rx, ry, rz, r
	cdef double cos_theta1, cos_theta2, cos_alpha

	# diagonal
	for i in range(N):
		r_arr[i,i] = 0.
		cth1_cth2_arr[i,i] = 0.
		sth1_sth2_arr[i,i] = 1.0

	# off-diagonal
	for i in range(N-1):
		for j in range(i+1,N):
			rx = chi[j]*nhat[j,0] - chi[i]*nhat[i,0]
			ry = chi[j]*nhat[j,1] - chi[i]*nhat[i,1]
			rz = chi[j]*nhat[j,2] - chi[i]*nhat[i,2]
			r = sqrt(rx*rx + ry*ry + rz*rz)
			r_arr[i,j] = r
			r_arr[j,i] = r

			if r < 1e-14: # theta1=pi/2 and theta2=pi/2
				cth1_cth2_arr[i,j] = 0.
				cth1_cth2_arr[j,i] = 0.
				sth1_sth2_arr[i,j] = 1.
				sth1_sth2_arr[j,i] = 1.
			else:
				cos_theta1 = (rx*nhat[i,0] + ry*nhat[i,1] + rz*nhat[i,2])/r
				cos_theta2 = (rx*nhat[j,0] + ry*nhat[j,1] + rz*nhat[j,2])/r
				cos_alpha = nhat[i,0]*nhat[j,0] + nhat[i,1]*nhat[j,1] + nhat[i,2]*nhat[j,2]
				cth1_cth2_arr[i,j] = cos_theta1*cos_theta2
				cth1_cth2_arr[j,i] = cth1_cth2_arr[i,j]
				sth1_sth2_arr[i,j] = cos_alpha - cth1_cth2_arr[i,j]
				sth1_sth2_arr[j,i] = sth1_sth2_arr[i,j]

	return np.asarray(r_arr), np.asarray(cth1_cth2_arr), np.asarray(sth1_sth2_arr)


