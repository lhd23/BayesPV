from __future__ import print_function
import numpy as np
import os
import sys
import emcee
from emcee.utils import MPIPool

import FP_mock as FP

dirname = os.path.dirname(os.path.abspath(__file__))

par_names = ['sig_r', 'sig_s', 'sig_i', 'rho1', 'rho2', 'rho3', 'rbar', 'sbar', 'ibar', 'sigma80', 'Om0']

def logpost(p):
	pd = FP.get_param_dict(par_names, p)
	lprior = FP.logprior(**pd)
	if not np.isfinite(lprior):
		return -np.inf
	return FPlike(p) + dzlike(p) + lprior


mock = FP.FP_mock(sig1=FP.SIG1*0.1, sig2=FP.SIG2*0.1, sig3=FP.SIG3*0.1)
# mock.init_6dF_like_data(fname='6dF_table.input', fname_pv='v.txt')
mock.init_6dF_like_data(N=1000)

Eyy = mock.Eyy # np.zeros((2*mock.N, 2*mock.N))
y = np.concatenate((mock.shat, mock.ihat))
FPlike = FP.FP_like(y=y, Eyy=Eyy, N=mock.N, par_names=par_names)
dzlike = FP.distz_like(y=y, Eyy=Eyy, N=mock.N, z=mock.ztot, RA=mock.RA, DEC=mock.DEC,
						theta=mock.angsize, par_names=par_names)

ndim = 11 # number of parameters
nwalkers = 100
nsamples = 800

print('Number of parameters: {}'.format(ndim))
print('Number of walkers: {}'.format(nwalkers))
print('Number of samples: {}'.format(nsamples))

p_true = np.array([FP.SIG_R*0.1, FP.SIG_S*0.1, FP.SIG_I*0.1, FP.RHO1, FP.RHO2, FP.RHO3,
					FP.RBAR, FP.SBAR, FP.IBAR, FP.sigma80, FP.Om0])
p_centre = np.tile(p_true, nwalkers)
dp = np.random.randn(ndim * nwalkers) / 500.
p = (p_centre*(1.0 + dp)).reshape((nwalkers, ndim))

pool = MPIPool()
if not pool.is_master():
	pool.wait()
	sys.exit(0)
# pool = None

sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, pool=pool)

for i, (p, lnprob, lnlike) in enumerate(sampler.sample(p, iterations=nsamples)):
	if (i+1) % 100 == 0:
		print("{0:5.1%}".format(float(i)/nsamples))
		np.savetxt('chain.txt', sampler.flatchain)

np.savetxt('chain.txt', sampler.flatchain)

# MCMC sampling complete

print('\nmean acceptance fraction: ', np.mean(sampler.acceptance_fraction))

if pool is not None:
	pool.close()
