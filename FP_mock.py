from __future__ import print_function
import numpy as np
from scipy import linalg
from scipy.stats import uniform, multivariate_normal, reciprocal, norm

import os

import cosmology
from fast_pairs import cy_pairs, cy_nhat

dirname = os.path.dirname(os.path.abspath(__file__))


def get_cov_FP(a, b, sig1, sig2, sig3):
	v1 = np.array([1,-a,-b])/np.sqrt(1+a**2+b**2)
	v2 = np.array([b,0,1])/np.sqrt(1+b**2)
	v3 = np.array([-a,-(1+b**2),a*b]) \
			/ np.sqrt(1.+b**2)/np.sqrt(1+a**2+b**2)
	O = np.column_stack((v1,v2,v3))
	D = np.diag([sig1**2, sig2**2, sig3**2])
	return np.matmul(O, np.matmul(D, O.T))

# Fundamental Plane parameters
a_FP = 1.502
b_FP = -0.877
c_FP = -0.303
SBAR = 2.188
IBAR = 3.184
RBAR = a_FP*SBAR + b_FP*IBAR +c_FP # 0.191008
SIG1 = 0.052
SIG2 = 0.315
SIG3 = 0.169

C_FP = get_cov_FP(a_FP, b_FP, SIG1, SIG2, SIG3)
SIG_R = np.sqrt(C_FP[0,0])
SIG_S = np.sqrt(C_FP[1,1])
SIG_I = np.sqrt(C_FP[2,2])
RHO1 = C_FP[0,1] / (SIG_R * SIG_S)
RHO2 = C_FP[0,2] / (SIG_R * SIG_I)
RHO3 = C_FP[1,2] / (SIG_S * SIG_I)

# Cosmological parameters
Om0 = 0.272
Ol0 = 1.0 - Om0
h = 0.704
ns = 0.961
sigma80 = 0.807


def get_cov_FP2(sig_r, sig_s, sig_i, rho1, rho2, rho3):
	cov = np.diag([sig_r**2, sig_s**2, sig_i**2])
	cov[0,1] = rho1 * sig_r * sig_s
	cov[0,2] = rho2 * sig_r * sig_i
	cov[1,2] = rho3 * sig_s * sig_i
	cov[1,0] = cov[0,1]
	cov[2,0] = cov[0,2]
	cov[2,1] = cov[1,2]
	return cov


def sample_6dF_positions(N=200, save_to_file=False, seed=None):
	"""
	This creates a file of a random sampling N objects from 6dF data.
	This is because we do not want to use all 6dF data
	(11,000 galaxies/clusters means huge matrices to invert)
	We only use spatial data from 6dF (z,RA,DEC specifies comoving pos)
	"""
	data = np.genfromtxt('{}/{}'.format(dirname,'campbell_table8.ascii'), skip_header=47)
	N_6dF = data.shape[0]
	RA = data[:,2] # degrees
	DEC = data[:,3] # degrees
	czcmb = data[:,4]
	# zcmb = czcmb / 2.99792458e5
	np.random.seed(seed)
	zcmb = np.random.uniform(low=0.006, high=0.02, size=N_6dF)
	inds = np.random.choice(np.arange(N_6dF), size=N, replace=False)
	print('Sampling 6dF angular coordinates')
	RA, DEC, zcmb = RA[inds], DEC[inds], zcmb[inds]
	if save_to_file:
		np.savetxt('6dF_table_N{}.input'.format(N), np.column_stack([RA, DEC, zcmb]), delimiter='   ')
		print('File "6dF_table_N{}.input" created'.format(N))
	return RA, DEC, zcmb


def sample_mock_phase_space(N=1000, seed=None):
	mockdata = np.genfromtxt('{}/{}'.format(dirname,'mock-pv2'))
	Ntot = mockdata.shape[0]
	zcmb = mockdata[:,1]
	RA = mockdata[:,3] # degrees
	DEC = mockdata[:,4] # degrees
	vlos = mockdata[:,5]

	np.random.seed()
	inds = np.random.choice(np.arange(Ntot), size=N, replace=False)
	sampled_data = np.column_stack([RA[inds], DEC[inds], zcmb[inds]])

	np.savetxt('pos.txt', sampled_data, delimiter='   ', fmt='%11.4e')
	np.savetxt('vel.txt', vlos[inds], delimiter='   ', fmt='%11.4e')


def get_param_dict(par_names, pars):
	X = zip(par_names, pars)
	return dict(X)


class FP_mock:

	def __init__(self, a=a_FP, b=b_FP, rbar=RBAR, sbar=SBAR, ibar=IBAR,
			sig1=SIG1, sig2=SIG2, sig3=SIG3,
			Om0=Om0, h=h, ns=ns, sigma80=sigma80, seed=None):

		#   Fundamental plane
		self.a = a
		self.b = b
		self.rbar = rbar
		self.sbar = sbar
		self.ibar = ibar
		self.sig1 = sig1
		self.sig2 = sig2
		self.sig3 = sig3

		#   Cosmological
		self.Om0 = Om0
		self.h = h
		self.ns = ns
		self.sigma80 = sigma80
		self.cosmo_model = cosmology.cosmo(Om0=self.Om0, h=self.h, ns=self.ns, sigma80=self.sigma80)

		self.cov_FP = get_cov_FP(self.a, self.b, self.sig1, self.sig2, self.sig3)
		self.mean_FP = np.array([self.rbar, self.sbar, self.ibar])
		self.mn = multivariate_normal(mean=self.mean_FP, cov=self.cov_FP)
		self.seed = seed # None is random state

		print('Data will be generated using parameters:', \
				'\na: ', self.a, '\nb: ', self.b, '\nrbar: ', self.rbar, \
				'\nsbar: ', self.sbar, '\nibar: ', self.ibar, \
				'\nsig1: ', sig1, '\nsig2: ', sig2, '\nsig3: ', sig3)

	@property
	def c(self):
                return self.rbar - (self.a*self.sbar + self.b*self.ibar)

	def _get_rsi(self, N):
		"""
		FP r,s,i are independent of position of galaxy.
		There are no restrictions from selection effects
		"""
		np.random.seed(self.seed)
		samps = self.mn.rvs(size=N) #shape=(N,3)
		r, s, i = samps[:,0], samps[:,1], samps[:,2]
		return (r,s,i)

	def _get_angsize(self, r, z, v=0.0):
		dA = np.zeros_like(r)
		for i in range(dA.size):
			dA[i] = self.cosmo_model.dA(z[i], v[i], use_lowz=True)
		Re = 10**r # effective size in units h^-1 kpc (see Magoulas+12)
		return Re/dA

	def _get_R_PV(self, coords, use_interpolation=True, use_lowz=True):
		"""
		Compute the covariance matrix of LOS velocities

		Parameters
		---------------------------
		coords: array-like
			list or array of 3-tuples (zcmb,RA,DEC)
		"""
		N = len(coords)
		R = np.zeros((N,N))
		for i in range(N):
			for j in range(i,N):
				if i == j:
					R[i,i] = self.cosmo_model.sigmav**2
				else:
					R[i,j] = self.cosmo_model.xiV(coords[i], coords[j], use_interpolation, use_lowz)
					R[j,i] = R[i,j]
		return R

	def _get_R_PV_fast(self, z, alpha, delta, sig_star=0.0):
		"""
		By default this function uses interpolation for correlation functions
		and also the low-z approximation to compute comoving distance

		Optional parameters
		-------------------
		sig_star: 1D incoherent velocity dispersion

		"""
		N = z.size
		n_hats_ = np.zeros((N,3))
		n_hats = cy_nhat(alpha, delta, n_hats_)

		chi_arr = self.cosmo_model.chi_lowz(z) * 1e-3 # units Mpc/h
		r_arr = np.zeros((N,N))
		cos1cos2_arr = np.zeros((N,N))
		sin1sin2_arr = np.zeros((N,N))
		r12, C12, S12 = cy_pairs(chi_arr, n_hats, r_arr, cos1cos2_arr, sin1sin2_arr)

		if self.cosmo_model.log_xi_perp_interpolator is None:
			self.cosmo_model.init_xiV_interpolation()

		iu = np.triu_indices(N, k=1)
		r12_offdiag_flat = r12[iu] # 1d array
		xi_perp_flat = 10**self.cosmo_model.log_xi_perp_interpolator(r12_offdiag_flat)
		xi_para_flat = self.cosmo_model.xi_para_interpolator(r12_offdiag_flat)
		xi_perp = np.zeros((N,N))
		xi_para = np.zeros((N,N))
		xi_perp[iu] = xi_perp_flat
		xi_para[iu] = xi_para_flat

		R = S12*xi_perp + C12*xi_para # elementwise multiplication
		R = R + R.T
		np.fill_diagonal(R, self.cosmo_model.sigmav**2 + sig_star**2)
		return R

	def add_noise_to_y(self, slatent, ilatent, s_percentage_err=2.0, i_percentage_err=2.0):
		print('Adding {}% errors to s and {}% errors to i'.format(s_percentage_err,i_percentage_err))
		if s_percentage_err == 0.0 and i_percentage_err == 0.0:
			shat = slatent
			ihat = ilatent
			Eyy = np.zeros((2*ihat.size, 2*ihat.size))
		else:
			s_err = slatent * (s_percentage_err/100.0)
			i_err = ilatent * (i_percentage_err/100.0)
			np.random.seed(self.seed)
			ds = np.array([norm.rvs(loc=0, scale=std) for std in s_err])
			di = np.array([norm.rvs(loc=0, scale=std) for std in i_err])
			shat = slatent + ds
			ihat = ilatent + di
			y_err = np.concatenate((s_err, i_err))
			Eyy = np.diag(y_err**2)
		return shat, ihat, Eyy

	def load_data_from_file(self, fname):
		shat, s_err, ihat, i_err, ztot, RA, DEC, angsize = np.loadtxt(fname, unpack=True)
		y_err = np.concatenate((s_err, i_err))
		Eyy = np.diag(y_err**2)
		return shat, ihat, Eyy, ztot, RA, DEC, angsize

	def init_6dF_like_data(self, N=500, load_data_fname=None, fname=None, fname_pv=None,
					save_to_file=True, s_percentage_err=0.0, i_percentage_err=0.0, scale=1.0):
		"""
		In order to initialize the likelihoods we need to either load from file or
		generate the following data:

			shat:	 N-dim. vector of log velocity dispersions
			ihat:  N-dim. vector of log velocity dispersions
			ztot:  N-dim. vector of observed redshifts
			angsize:  N-dim. vector of observed angular sizes
			Eyy:  Error matrix for data vector y=(shat,ihat)
			R:  Peculiar velocity covariance matrix

		N is the number of galaxies
		"""
		if load_data_fname is not None:
			print('Reading mock data from {}'.format(load_data_fname))
			self.shat, self.ihat, self.Eyy, self.ztot, self.RA, self.DEC, self.angsize = \
						self.load_data_from_file(load_data_fname)
			self.N = self.shat.shape[0]
		else:
			np.random.seed(self.seed)
			if fname is None:
				self.N = N
				self.RA, self.DEC, self.zcmb = sample_6dF_positions(self.N, seed=self.seed)
			else:
				print('Reading position data from {}'.format(fname))
				data = np.loadtxt('{}/{}'.format(dirname,fname))
				self.N = data.shape[0]
				self.RA = data[:,0]
				self.DEC = data[:,1]
				self.zcmb = data[:,2]
			self.alpha = self.RA * np.pi/180
			self.delta = self.DEC * np.pi/180

			print('Generating rsi data for {} galaxies...'.format(self.N))
			rlatent, slatent, ilatent = self._get_rsi(self.N)
			self.shat, self.ihat, self.Eyy = self.add_noise_to_y(slatent, ilatent, \
									s_percentage_err, i_percentage_err)

			if fname_pv is not None:
				print('Reading PVs from {}...'.format(fname_pv))
				self.V_samp = np.loadtxt('{}/{}'.format(dirname,fname_pv)) * np.sqrt(scale)
				self.ztot = self.cosmo_model.ztot(self.zcmb, self.V_samp)
				self.angsize = self._get_angsize(rlatent, self.ztot, self.V_samp)
			else:
				print('Generating PVs for {} galaxies...'.format(self.N))
				self.cosmo_model.init_xiV_interpolation()
				# list_of_coords = [coords for coords in zip(self.zcmb, self.RA, self.DEC)]
				# R = self._get_R_PV(list_of_coords)
				R = self._get_R_PV_fast(self.zcmb, self.alpha, self.delta)
				R *= scale

				# np.random.seed(self.seed)
				# mn_pv = multivariate_normal(mean=np.zeros(self.N), cov=R, allow_singular=True)

				#   Keep generating samples until all angsize is positive definite
				is_unphysical = True
				while is_unphysical:
					# self.V_samp = mn_pv.rvs()
					self.V_samp = np.random.multivariate_normal(mean=np.zeros(self.N), cov=R)
					self.ztot = self.cosmo_model.ztot(self.zcmb, self.V_samp)
					self.angsize = self._get_angsize(rlatent, self.ztot, self.V_samp)
					if np.all(self.angsize > 0.0):
						is_unphysical = False

				if save_to_file:
					print('Randomly generated and all other data saved to files')
					data_rand = np.column_stack([self.zcmb, self.RA, self.DEC, \
									slatent, ilatent, self.V_samp])
					filename = 'latent_data_N{}.txt'.format(self.N)
					np.savetxt(filename, data_rand, delimiter='  ', fmt='%11.4e')

					s_err = np.sqrt(np.diag(self.Eyy)[:self.N])
					i_err = np.sqrt(np.diag(self.Eyy)[self.N:])
					data_mock = np.column_stack([self.shat, s_err, self.ihat, i_err, \
									self.ztot, self.RA, self.DEC, self.angsize])
					filename = 'mock_sample_N{}.txt'.format(self.N)
					np.savetxt(filename, data_mock, delimiter='  ', fmt='%11.4e')


	def data_dict(self): # these are things we actually measure (observables)
		return {'z': self.ztot, 'theta': self.angsize, 's': self.shat,
			'i': self.ihat, 'RA': self.RA, 'DEC': self.DEC, 'Eyy': self.Eyy}

	def fit(self, scale=0.1):
		"""
		Fitting FP and the amplitude of the power spectrum only. We	add a
		log uniform prior on the scale parameter sigma80. Run this method
		only after initialising data.
		"""
		def _negloglike(sig_r,sig_s,sig_i,rho1,rho2,rho3,rbar,sbar,ibar,sigma80,Om0,sig_star):
			p = [sig_r,sig_s,sig_i,rho1,rho2,rho3,rbar,sbar,ibar,sigma80,Om0,sig_star]
			pd = get_param_dict(par_names, p)
			return FPlike.negloglike(p) + dzlike.negloglike(p) - logprior(**pd)

		try:
			import iminuit
		except:
			raise ImportError('Module iminuit not found')

		par_names = ['sig_r', 'sig_s', 'sig_i', 'rho1', 'rho2', 'rho3',
					'rbar', 'sbar', 'ibar', 'sigma80', 'Om0', 'sig_star']
		Eyy = self.Eyy # np.zeros((2*self.N, 2*self.N))
		y = np.concatenate((self.shat, self.ihat)) # shape=(2*N,)
		FPlike = FP_like(y=y, Eyy=Eyy, N=self.N, par_names=par_names)
		dzlike = distz_like(y=y, Eyy=Eyy, N=self.N, z=self.ztot, RA=self.RA, DEC=self.DEC,
						theta=self.angsize, par_names=par_names)

		self.m = iminuit.Minuit(_negloglike,
					sig_r=SIG_R*scale, sig_s=SIG_S*scale, sig_i=SIG_I*scale,
					rho1=RHO1, rho2=RHO2, rho3=RHO3,
					rbar=RBAR, sbar=SBAR, ibar=IBAR,
					error_sig_r=0.1*scale, error_sig_s=0.1*scale, error_sig_i=0.01*scale,
					error_rho1=0.01, error_rho2=0.01, error_rho3=0.01,
					error_rbar=0.01, error_sbar=0.2, error_ibar=0.3,
					limit_sig_r=(1e-6,None), limit_sig_s=(1e-5,None), limit_sig_i=(1e-5,None),
					limit_rho1=(-0.999,0.999), limit_rho2=(-0.999,0.999), limit_rho3=(-0.999,0.999),
					limit_rbar=(None,None), limit_sbar=(None,None), limit_ibar=(None,None),
					sigma80=0.81,
					error_sigma80=0.01,
					limit_sigma80=(1e-4,None),
					Om0=0.27,
					error_Om0=0.01,
					limit_Om0=(0.01,0.99),
					errordef=0.5,
					print_level=1,
					fix_sig_r=True,
					fix_sig_s=True,
					fix_sig_i=True,
					fix_rho1=True,
					fix_rho2=True,
					fix_rho3=True,
					fix_rbar=True,
					fix_sbar=True,
					fix_ibar=True,
					fix_sig_star=False,
					sig_star=350.,
					error_sig_star=10.,
					limit_sig_star=(1e-2,None)
					)

		self.m.migrad()
		if self.m.migrad_ok():
			print('MIGRAD optimization success')
			self.FCN = self.m.get_fmin()['fval']
			self.pars = self.m.np_values()
			self.log_FPlike_max = -FPlike.negloglike(self.pars)
			self.log_dzlike_max = -dzlike.negloglike(self.pars)
			self.sigma80_val = self.m.values['sigma80']
			self.sigma80_err = self.m.errors['sigma80']
			# self.m.print_matrix() # correlation table
			self.m.print_param()
			print('dz like max: ', self.log_dzlike_max)
		else:
			print('MIGRAD optimization failure')

	def estimate_V(self, pars, plot_eig=False): # call only after data initialised
		par_names = ['sig_r', 'sig_s', 'sig_i', 'rho1', 'rho2', 'rho3',
                             'rbar', 'sbar', 'ibar', 'sigma80', 'Om0']
		y = np.concatenate((self.shat, self.ihat)) # shape=(2*N,)
		dzlike = distz_like(y=y, Eyy=self.Eyy, N=self.N, z=self.ztot,
						RA=self.RA, DEC=self.DEC, theta=self.angsize,
						par_names=par_names)
		V, Cov_V, rho_V, out = dzlike.V_MAP(pars, plot_eig=plot_eig)
		err_V = np.sqrt(np.diag(Cov_V))

		u, s, _ = np.linalg.svd(Cov_V)
		v0 = u[:,-1]
		v1 = u[:,-2]
		v2 = u[:,-3]
		v3 = u[:,-4]
		idx0 = np.argmin(np.abs(v0))
		idx1 = np.argmin(np.abs(v1))
		idx2 = np.argmin(np.abs(v2))
		idx3 = np.argmin(np.abs(v3))
		print('max compt (index,value): ', idx0, v0[idx0])
		print('max compt (index,value): ', idx1, v1[idx1])
		print('max compt (index,value): ', idx2, v1[idx2])
		print('max compt (index,value): ', idx3, v1[idx3])
		return V, err_V, rho_V, out


class loglike_gaussian(object):

	def __init__(self, x=None, y=None, cov=None, N=None, par_names=None):
		self._x = x
		self._y = y
		self._cov = cov # statistical errors
		self.N = N
		self.par_names = par_names

	@property
	def x(self):
		return self._x

	@property
	def y(self):
		return self._y

	@property
	def cov(self):
		return self._cov

	def __call__(self, pars):
		pars = np.asarray(pars)
		return self._loglike(pars)

	def neg2loglike(self, pars): #chi2 + ln det C
		return -2.0 * self.__call__(pars)

	def negloglike(self, pars):
		return -self.__call__(pars)

	def _loglike(self, pars):
		raise NotImplementedError('_loglike not implemented')


class FP_like(loglike_gaussian):

	def __init__(self, y=None, Eyy=None, N=None, par_names=None, fixed_par_dict={}):
		loglike_gaussian.__init__(self, y=y, cov=Eyy, N=N, par_names=par_names)
		shat = self.y[:self.N]
		ihat = self.y[self.N:]
		self.ym = np.column_stack((shat,ihat)) # shape=(N,2)
		self.Em = np.zeros((self.N,2,2))
		for m in range(self.N):
			self.Em[m,0,0] = self.cov[m,m]
			self.Em[m,1,1] = self.cov[self.N+m,self.N+m]
			self.Em[m,0,1] = self.cov[m,self.N+m]
			self.Em[m,1,0] = self.Em[m,0,1]

		all_names = ['sig_r', 'sig_s', 'sig_i', 'rho1', 'rho2', 'rho3',
				'rbar', 'sbar', 'ibar', 'sigma80', 'Om0']

		# Fixed calibration parameters are specified by fixed_par_dict
		if bool(fixed_par_dict): # evals to False if empty
			fix_par_names = list(fixed_par_dict.keys())
			my_par_names = par_names + fix_par_names
			if sorted(my_par_names) != sorted(all_names):
				raise ValueError('Fixed parameters not specified or overlap with free ones')
		self.fixed_par_dict = fixed_par_dict

	def get_param_dict(self, pars):
		param_dict = dict(zip(self.par_names, pars))
		param_dict.update(self.fixed_par_dict)
		return param_dict

	def _loglike(self, pars):
		pd = self.get_param_dict(pars)
		cov_FP = get_cov_FP2(pd['sig_r'], pd['sig_s'], pd['sig_i'], pd['rho1'], pd['rho2'], pd['rho3'])
		mu = np.array([pd['sbar'], pd['ibar']])
		C_2x2 = cov_FP[1:,1:]

		ret = 0.0
		if np.allclose(self.Em, 0.*self.Em): # then need only compute Cinv and det once
			C = C_2x2
			det = C[0,0]*C[1,1] - C[0,1]*C[1,0]
			Cinv = np.array([[C[1,1], -C[0,1]], [-C[0,1], C[0,0]]]) / det
			for m in range(self.N):
				res = self.ym[m] - mu
				chi2 = np.dot(res, np.dot(Cinv, res))
				ret += -0.5*chi2
			ret += self.N * (-0.5*np.log(det))
			ret += self.N * (-0.5*2.*np.log(2.*np.pi))
		else:
			for m in range(self.N):
				res = self.ym[m] - mu
				CpE = C_2x2 + self.Em[m]
				det = CpE[0,0]*CpE[1,1] - CpE[0,1]*CpE[1,0]
				Cinv = np.array([[CpE[1,1], -CpE[0,1]], [-CpE[0,1], CpE[0,0]]]) / det
				chi2 = np.dot(res, np.dot(Cinv, res))
				ret += -0.5 * (chi2 + np.log(det) + 2.*np.log(2.*np.pi))

		if np.isnan(ret):
			return float(-1e20)
		else:
			return ret

class distz_like(loglike_gaussian):

	def __init__(self, y=None, Eyy=None, N=None, z=None,
				RA=None, DEC=None, theta=None, par_names=None, fixed_par_dict={}):
		loglike_gaussian.__init__(self, y=y, cov=Eyy, N=N, par_names=par_names)
		self._z = z # observed (total) redshift
		self._ltheta = np.log10(theta)
		self._coords = [x for x in zip(z, RA, DEC)]

		alpha = RA * np.pi/180
		delta = DEC * np.pi/180
		n_hats = np.zeros((self.N,3))
		self.n_hats = cy_nhat(alpha, delta, n_hats)

		all_names = ['sig_r', 'sig_s', 'sig_i', 'rho1', 'rho2', 'rho3',
				'rbar', 'sbar', 'ibar', 'sigma80', 'Om0', 'sig_star']

		# Fixed calibration parameters are specified by fixed_par_dict
		if bool(fixed_par_dict): # evals to False if empty
			fix_par_names = list(fixed_par_dict.keys())
			my_par_names = par_names + fix_par_names
			if sorted(my_par_names) != sorted(all_names):
				raise ValueError('Fixed parameters not specified or overlap with free ones')
		self.fixed_par_dict = fixed_par_dict

	@property
	def z(self): # observed redshift
		return self._z

	@property
	def ltheta(self):
		return self._ltheta

	@property
	def coords(self): # list of coordinates as 3-tuples (z,RA,DEC)
		return self._coords

	def get_param_dict(self, pars):
		param_dict = dict(zip(self.par_names, pars))
		param_dict.update(self.fixed_par_dict)
		return param_dict

	def _loglike(self, pars):
		"""
		The inversion of the covariance matrix C is done via Cholesky
		decomposition, which is generally faster for large matrices:
		    C = LL^T, where L is lower triangle matrix.

		Instead of computing C^{-1} the system of eqns Cu = y-mu is
		solved for x = C^{-1} (y-mu). The 'chi squared' part is then
		(y-mu)^T C^{-1} (y-mu) = (y-mu)^T x

		Determinant of lower tri is given by prod of diag entries
		det(C) = det(LL^T) = det(L)^2 = prod(diag(L))^2

		Returns
		-------
		ln(likelihood), where ln is the natural log.
		"""

		def _ybar(sbar, ibar, **kwargs):
			sbar_arr = sbar * np.ones(self.N)
			ibar_arr = ibar * np.ones(self.N)
			return np.concatenate((sbar_arr, ibar_arr))

		def _Crr(sig_r, sig_s, sig_i, rho1, rho2, rho3, **kwargs):
			cov_FP = get_cov_FP2(sig_r, sig_s, sig_i, rho1, rho2, rho3) # 3x3
			return np.diag(self.N*[cov_FP[0,0]])

		def _Cry(sig_r, sig_s, sig_i, rho1, rho2, rho3, **kwargs):
			cov_FP = get_cov_FP2(sig_r, sig_s, sig_i, rho1, rho2, rho3) # 3x3
			v = cov_FP[1:3,0].reshape(-1, 1) # shape = (2,1)
			return linalg.kron(v, np.eye(self.N))

		def _Cyy(sig_r, sig_s, sig_i, rho1, rho2, rho3, **kwargs):
			cov_FP = get_cov_FP2(sig_r, sig_s, sig_i, rho1, rho2, rho3) # 3x3
			return linalg.kron(cov_FP[1:3,1:3], np.eye(self.N))

		def _A():
			A = np.zeros(self.N)
			for i,z in enumerate(self.z):
				kappa = cosmo_model.kappa_v(z, v=1.0) # only need factor so set v=1
				A[i] = -kappa/np.log(10.0)
			return np.diag(A)

		def _R(use_interpolation=True, use_lowz=True):
			R = np.zeros((self.N,self.N))
			for i in range(self.N):
				for j in range(i,self.N):
					if i == j:
						R[i,i] = cosmo_model.sigmav**2
					else:
						R[i,j] = cosmo_model.xiV(self.coords[i], self.coords[j],
									use_interpolation, use_lowz)
						R[j,i] = R[i,j]
			return R

		def _R_fast(sig_star=350.0, b1=1., **kwargs):
			"""
			To create the covariance this method instead pushes the loops down
			to cython code. The covariance is written as the sum of two terms,
			each made up of a matrix product consisting of the angular part and
			correlation part.

			By default this function uses interpolation for correlation functions
			and also the low-z approximation to compute comoving distance
			"""
			chi_arr = cosmo_model.chi_lowz(self.z) * 1e-3 # units Mpc/h
			r_arr = np.zeros((self.N,self.N))
			cos1cos2_arr = np.zeros((self.N,self.N))
			sin1sin2_arr = np.zeros((self.N,self.N))
			r12, C12, S12 = cy_pairs(chi_arr, self.n_hats, r_arr, cos1cos2_arr, sin1sin2_arr)

			if cosmo_model.log_xi_perp_interpolator is None:
				cosmo_model.init_xiV_interpolation()

			iu = np.triu_indices(self.N, k=1)
			r12_offdiag_flat = r12[iu] # 1d array
			xi_perp_flat = 10**cosmo_model.log_xi_perp_interpolator(r12_offdiag_flat)
			xi_para_flat = cosmo_model.xi_para_interpolator(r12_offdiag_flat)
			xi_perp = np.zeros((self.N,self.N))
			xi_para = np.zeros((self.N,self.N))
			xi_perp[iu] = xi_perp_flat
			xi_para[iu] = xi_para_flat

			R = (S12 * xi_perp) + (C12 * xi_para) # elementwise multiplication
			R = R + R.T
			np.fill_diagonal(R, cosmo_model.sigmav**2 + sig_star**2)
			# R *= b1**2 # linear bias
			return R

		pd = self.get_param_dict(pars)

		#   If varying Omegam matter uncomment below
		cosmo_model = cosmology.cosmo(**pd)
		cosmo_model.init_xiV_interpolation()
		dA = np.zeros(self.N)
		for i,z in enumerate(self.z):
			dA[i] = cosmo_model.dA(z, use_lowz=True)
		ldA = np.log10(dA)
		A = _A()
		R = _R_fast(**pd)
		ARA = np.matmul(A, np.matmul(R, A))

		Crr = _Crr(**pd)
		Cry = _Cry(**pd)
		Cyy = _Cyy(**pd)
		try:
			# Cyy_tot_inv = linalg.inv(Cyy + self.cov)
			Cyy_tot_cholfac = linalg.cho_factor(Cyy+self.cov, overwrite_a=True, lower=True)
		except ValueError:
			return float(-1e20)
		# Ctot = ARA + Crr - np.matmul(Cry.T, np.matmul(Cyy_tot_inv, Cry))
		Ctot = ARA + Crr - np.dot(Cry.T, linalg.cho_solve(Cyy_tot_cholfac, Cry))

		dy = _ybar(**pd) - self.y
		# rbar_shift = pd['rbar'] - np.matmul(Cry.T, np.matmul(Cyy_tot_inv, dy))
		rbar_shift = pd['rbar'] - np.dot(Cry.T, linalg.cho_solve(Cyy_tot_cholfac, dy))
		Delta = self.ltheta + ldA - rbar_shift

		if np.any(np.isnan(Delta)):
			return float(-1e20)
		try:
			chol_fac = linalg.cho_factor(Ctot, overwrite_a=True, lower=True)
			chi2 = np.dot(Delta, linalg.cho_solve(chol_fac, Delta))
			lndet = 2. * np.sum(np.log(np.diag(chol_fac[0])))
			ret = -0.5 * (chi2 + self.N*np.log(2.*np.pi) + lndet)

			if not np.isfinite(ret):
				return float(-1e20)
			else:
				return ret
		except linalg.LinAlgError: # when not positive definite
			return float(-1e20)
		except ValueError:
			return float(-1e20)

	def V_MAP(self, pars, option=2, plot_eig=False):
		def _ybar(sbar, ibar, **kwargs):
			sbar_arr = sbar * np.ones(self.N)
			ibar_arr = ibar * np.ones(self.N)
			return np.concatenate((sbar_arr, ibar_arr))

		def _Crr(sig_r, sig_s, sig_i, rho1, rho2, rho3, **kwargs):
			cov_FP = get_cov_FP2(sig_r, sig_s, sig_i, rho1, rho2, rho3) # 3x3
			return np.diag(self.N*[cov_FP[0,0]])

		def _Cry(sig_r, sig_s, sig_i, rho1, rho2, rho3, **kwargs):
			cov_FP = get_cov_FP2(sig_r, sig_s, sig_i, rho1, rho2, rho3) # 3x3
			v = cov_FP[1:3,0].reshape(-1, 1) # shape = (2,1)
			return linalg.kron(v, np.eye(self.N))

		def _Cyy(sig_r, sig_s, sig_i, rho1, rho2, rho3, **kwargs):
			cov_FP = get_cov_FP2(sig_r, sig_s, sig_i, rho1, rho2, rho3) # 3x3
			return linalg.kron(cov_FP[1:3,1:3], np.eye(self.N))

		def _A_inv():
			A = np.zeros(self.N)
			for i,z in enumerate(self.z):
				kappa = cosmo_model.kappa_v(z, v=1.0) # only need factor so set v=1
				A[i] = -kappa/np.log(10.0)
			return np.diag(1./A)

		def _R_fast(sig_star=362.8, b1=1., **kwargs):
			"""
			To create the covariance this method instead pushes the loops down
			to cython code. The covariance is written as the sum of two terms,
			each made up of a matrix product consisting of the angular part and
			correlation part.

			By default this function uses interpolation for correlation functions
			and also the low-z approximation to compute comoving distance
			"""
			chi_arr = cosmo_model.chi_lowz(self.z) * 1e-3 # units Mpc/h
			r_arr = np.zeros((self.N,self.N))
			cos1cos2_arr = np.zeros((self.N,self.N))
			sin1sin2_arr = np.zeros((self.N,self.N))
			r12, C12, S12 = cy_pairs(chi_arr, self.n_hats, r_arr, cos1cos2_arr, sin1sin2_arr)

			if cosmo_model.log_xi_perp_interpolator is None:
				cosmo_model.init_xiV_interpolation()

			iu = np.triu_indices(self.N, k=1)
			r12_offdiag_flat = r12[iu] # 1d array
			xi_perp_flat = 10**cosmo_model.log_xi_perp_interpolator(r12_offdiag_flat)
			xi_para_flat = cosmo_model.xi_para_interpolator(r12_offdiag_flat)
			xi_perp = np.zeros((self.N,self.N))
			xi_para = np.zeros((self.N,self.N))
			xi_perp[iu] = xi_perp_flat
			xi_para[iu] = xi_para_flat

			R = (S12 * xi_perp) + (C12 * xi_para) # elementwise multiplication
			R = R + R.T
			np.fill_diagonal(R, cosmo_model.sigmav**2 + sig_star**2)
			# R *= b1**2 # linear bias
			return R

		pd = get_param_dict(self.par_names, pars)
		cosmo_model = cosmology.cosmo(**pd)
		cosmo_model.init_xiV_interpolation()
		Ainv = _A_inv()
		R = _R_fast()
		Crr = _Crr(**pd)
		Cry = _Cry(**pd)
		Cyy = _Cyy(**pd)
		Cyy_tot_inv = linalg.inv(Cyy + self.cov)
		try:
			# Cyy_tot_inv = linalg.inv(Cyy + self.cov)
			Cyy_tot_cholfac = linalg.cho_factor(Cyy+self.cov, overwrite_a=True, lower=True)
		except:
			raise ValueError

		Crr_prime = Crr - np.matmul(Cry.T, linalg.cho_solve(Cyy_tot_cholfac, Cry)) # Sig_0
		Noise = np.matmul(Ainv, np.matmul(Crr_prime, Ainv))
		W = np.matmul(R, linalg.inv(R+Noise)) # Wiener filter

		cosmo_model = cosmology.cosmo(**pd)
		cosmo_model.init_xiV_interpolation()
		dA = np.zeros(self.N)
		for i,z in enumerate(self.z):
			dA[i] = cosmo_model.dA(z, use_lowz=True)
		ldA = np.log10(dA)
		dy = _ybar(**pd) - self.y
		rbar_shift = pd['rbar'] - np.matmul(Cry.T, linalg.cho_solve(Cyy_tot_cholfac, dy))
		Delta = rbar_shift - (self.ltheta + ldA)
		V = np.matmul(W, np.matmul(Ainv,Delta))

		Hess = linalg.inv(R) + linalg.inv(Noise)
		Cov_V = linalg.inv(Hess)

		if plot_eig:
			u, s, _ = np.linalg.svd(W)

			idx = [np.argmax(np.abs(u[:,i])) for i in range(self.N)]
			vals = [u[j,i] for (j,i) in zip(idx,range(self.N))]
			print(vals)
			print(np.sort(idx))
			print(u[:,0])

			import matplotlib.pyplot as plt
			plt.plot(s, c='k', ls='None', marker='.')
			# plt.savefig('test.pdf')
			plt.show()

		if option == 1: # LSS correlations by highest to lowest
			rho_V = R / cosmo_model.sigmav**2
			B = rho_V
		elif option == 2: # total covariance
			B = Cov_V
		elif option == 3: # total correlations
			rho_V = np.zeros_like(Cov_V)
			for i in range(self.N):
				for j in range(i,self.N):
					if i == j:
						rho_V[i,i] = 1.
					else:
						rho_V[i,j] = Cov_V[i,j] / np.sqrt(Cov_V[i,i] * Cov_V[j,j])
						rho_V[j,i] = rho_V[i,j]
			B = rho_V
		elif option == 4:
			B = Hess

		r_, c_ = np.triu_indices(B.shape[1], 1) # row, column indices of upper triangle
		idx = B[r_,c_].argsort()[::-1] # high to low
		r,c = r_[idx], c_[idx]
		out = zip(r,c,B[r,c])
		return V, Cov_V, B, out


class classic_pv_likelihood(loglike_gaussian):

	def __init__(self, y=None, N=None, z=None, RA=None, DEC=None, par_names=None):
		loglike_gaussian.__init__(self, y=y, N=N, par_names=par_names)
		self._z = z # observed (total) redshift
		self._V = y # radial peculiar velocities

		alpha = RA * np.pi/180
		delta = DEC * np.pi/180
		n_hats = np.zeros((self.N,3))
		self.n_hats = cy_nhat(alpha, delta, n_hats)

	@property
	def z(self): # observed redshift
		return self._z

	@property
	def V(self):
		return self._V

	def get_param_dict(self, pars):
		param_dict = dict(zip(self.par_names, pars))
		return param_dict

	def _loglike(self, pars):
		"""
		Returns
		-------
		ln(likelihood), where ln is the natural log.
		"""
		def _R_fast(sig_star=350.0, b1=1., **kwargs):
			"""
			To create the covariance this method instead pushes the loops down
			to cython code. The covariance is written as the sum of two terms,
			each made up of a matrix product consisting of the angular part and
			correlation part.

			By default this function uses interpolation for correlation functions
			and also the low-z approximation to compute comoving distance
			"""
			chi_arr = cosmo_model.chi_lowz(self.z) * 1e-3 # units Mpc/h
			r_arr = np.zeros((self.N,self.N))
			cos1cos2_arr = np.zeros((self.N,self.N))
			sin1sin2_arr = np.zeros((self.N,self.N))
			r12, C12, S12 = cy_pairs(chi_arr, self.n_hats, r_arr, cos1cos2_arr, sin1sin2_arr)

			if cosmo_model.log_xi_perp_interpolator is None:
				cosmo_model.init_xiV_interpolation()

			iu = np.triu_indices(self.N, k=1)
			r12_offdiag_flat = r12[iu] # 1d array
			xi_perp_flat = 10**cosmo_model.log_xi_perp_interpolator(r12_offdiag_flat)
			xi_para_flat = cosmo_model.xi_para_interpolator(r12_offdiag_flat)
			xi_perp = np.zeros((self.N,self.N))
			xi_para = np.zeros((self.N,self.N))
			xi_perp[iu] = xi_perp_flat
			xi_para[iu] = xi_para_flat

			R = (S12 * xi_perp) + (C12 * xi_para) # elementwise multiplication
			R = R + R.T
			np.fill_diagonal(R, cosmo_model.sigmav**2 + sig_star**2)
			# R *= b1**2 # linear bias
			return R

		pd = self.get_param_dict(pars)

		#   If varying Omegam matter uncomment below
		cosmo_model = cosmology.cosmo(**pd)
		cosmo_model.init_xiV_interpolation()
		R = _R_fast(**pd)
		try:
			chol_fac = linalg.cho_factor(R, overwrite_a=True, lower=True)
			chi2 = np.dot(self.V, linalg.cho_solve(chol_fac, self.V))
			lndet = 2. * np.sum(np.log(np.diag(chol_fac[0])))
			ret = -0.5 * (chi2 + self.N*np.log(2.*np.pi) + lndet)

			if not np.isfinite(ret):
				return float(-1e20)
			else:
				return ret
		except linalg.LinAlgError: # when not positive definite
			return float(-1e20)
		except ValueError:
			return float(-1e20)


def logprior(sig_r=None, sig_s=None, sig_i=None, rho1=None, rho2=None, rho3=None,
             rbar=None, sbar=None, ibar=None,
	     sigma80=None, Om0=None, Om_a=0.01, Om_b=0.99, sig_star=None, **kwargs):

	ret = 0.0
	#   Uniform priors (location parameters)
	if rho1 is not None:
		ret += uniform.logpdf(rho1, loc=-0.999, scale=1.998)
	if rho2 is not None:
		ret += uniform.logpdf(rho2, loc=-0.999, scale=1.998)
	if rho3 is not None:
		ret += uniform.logpdf(rho3, loc=-0.999, scale=1.998)

	if rbar is not None:
		ret += uniform.logpdf(rbar, loc=0.05, scale=0.25)
	if sbar is not None:
		ret += 0.0 # uniform.logpdf(sbar, loc=0.5, scale=5.0)
	if ibar is not None:
		ret += 0.0 # uniform.logpdf(ibar, loc=1.0, scale=6.0)

	if Om0 is not None:
		ret += uniform.logpdf(Om0, loc=Om_a, scale=Om_b-Om_a)

	#   Log Uniform priors (scale parameters)
	if sig_r is not None:
		ret += reciprocal.logpdf(sig_r, 1e-05, 1e+01)
	if sig_s is not None:
		ret += reciprocal.logpdf(sig_s, 1e-05, 1e+01)
	if sig_i is not None:
		ret += reciprocal.logpdf(sig_i, 1e-05, 1e+01)
	if sigma80 is not None:
		ret += reciprocal.logpdf(sigma80, 1e-05, 1e+01)
	if sig_star is not None:
		ret += reciprocal.logpdf(sig_star, 1e-02, 600.)
	return ret


def main():

	# mock = FP_mock(sig1=0.1*SIG1, sig2=0.1*SIG2, sig3=0.1*SIG3)
	# mock.init_6dF_like_data(fname_pv='v.txt', save_to_file=False)
	# mock.fit()

	# for i in range(1):
	# 	seed = np.random.randint(low=1e5)
	# 	print('\nseed: ', seed)
	# 	mock = FP_mock(sig1=0.1*SIG1, sig2=0.1*SIG2, sig3=0.1*SIG3, seed=seed)
	# 	mock.init_6dF_like_data(N=1000, save_to_file=False)
	# 	mock.fit()

	mock = FP_mock(sig1=0.1*SIG1, sig2=0.1*SIG2, sig3=0.1*SIG3, seed=12345)
	mock.init_6dF_like_data(fname='pos_guo2.txt', fname_pv='v_guo2.txt')
	mock.fit()

	# seed = np.random.randint(low=1e5)
	# print('seed: ', seed)
	# mock = FP_mock(sig1=0.1*SIG1, sig2=0.1*SIG2, sig3=0.1*SIG3, seed=seed)
	# sample_mock_phase_space(N=1000, seed=seed)
	# mock.init_6dF_like_data(fname='pos.txt', fname_pv='vel.txt')
	# mock.fit()


	# mock = FP_mock(sig1=0.1*SIG1, sig2=0.1*SIG2, sig3=0.1*SIG3, seed=12345)
	# mock.init_6dF_like_data(load_data_fname='mock_sample_N1000.txt')
	# mock.fit()

if __name__ == '__main__':
	main()
