import numpy as np
from scipy import integrate
from scipy import interpolate


# Cosmological parameters
Om0 = 0.272
Ol0 = 1.0 - Om0
h = 0.704
ns = 0.961
sigma80 = 0.807

SPEEDOFLIGHT_KMS = 2.99792458e5


def nhat(alpha, delta):
	nhat = np.zeros(3)
	nhat[0] = np.cos(delta) * np.cos(alpha)
	nhat[1] = np.cos(delta) * np.sin(alpha)
	nhat[2] = np.sin(delta)
	return nhat

def angsep(alpha1, alpha2, delta1, delta2):
	cos_ang = np.sin(delta1)*np.sin(delta2) + np.cos(delta1)*np.cos(delta2)*np.cos(alpha1-alpha2)
	ang = np.arccos(cos_ang)
	return ang


class cosmo:

	def __init__(self, Om0=Om0, h=h, ns=ns, sigma80=sigma80, **kwargs):
		self.Om0 = Om0
		self.Ol0 = 1.0 - self.Om0
		self.Ob0 = 0.045
		self.Tcmb0 = 2.7255
		self.h = h
		self.ns = ns
		self.sigma80 = sigma80
		self.H0 = 100.0 # [h km/s/Mpc]
		self.q0 = 0.5*self.Om0 - self.Ol0
		self.gamma = 0.55 # growth index
		self._As = None
		self._sigmav = None
		self.log_xi_perp_interpolator = None
		self.xi_para_interpolator = None

		# Eisenstein & Hu (1998) zero baryon transfer function parameters
		ombom0 = self.Ob0 / self.Om0 # shorthand
		om0h2 = self.Om0 * self.h**2
		ombh2 = self.Ob0 * self.h**2
		self.theta2p7 = self.Tcmb0 / 2.7
		# Equation 31
		alphaGamma = 1.0 - 0.328*np.log(431.0*om0h2)*ombom0 + 0.38*np.log(22.3*om0h2)*ombom0**2
		# Quantities for Equation 30 (computed in transferFunction)
		self.Gamma1 = self.Om0*self.h*alphaGamma
		self.Gamma2 = self.Om0*self.h*(1.0-alphaGamma)
		# Equation 26
		self.s_EH98 = 44.5*np.log(9.83/om0h2) / np.sqrt(1.0+10.0*ombh2**0.75)

		# halofit spectral parameters
		self.rknl = None
		self.rneff = None
		self.rncur = None


	@property
	def dH(self):
		return (SPEEDOFLIGHT_KMS)/self.H0 * 1e3 # c/H_0 [h^-1 kpc]

	def E_Hub(self, z):
		"""
		Computes E(z) = H(z)/H0
		"""
		E2 = self.Om0*(1.+z)**3 + self.Ol0
		if np.all(E2 > 0.0):
			return np.sqrt(E2)
		else:
			return np.NaN

	def Omega_m(self, z):
		"""
		Evolution of omega matter with redshift
		"""
		EH = self.E_Hub(z)
		return self.Om0*(1.+z)**3 / EH**2

	def Omega_v(self, z):
		"""
		Evolution of omega vacuum with redshift
		"""
		EH = self.E_Hub(z)
		return self.Ol0 / EH**2

	def chi(self, z, use_lowz=False):
		"""
		Computes the comoving distance in units h^-1 kpc
		"""

		def _integrand(z):
			return 1.0/self.E_Hub(z) # 1/E(z) = H0/H(z)

		if use_lowz: # if z<<1
			return self.dH * (z - 0.5*(1.+self.q0)*z**2)
		else:
			if np.isclose(z, 0.0):
				return 0.0
			zp1 = z + 1.0
			if np.isfinite(_integrand(z)): # prevent negative square roots
				if np.isclose(self.Om0, 1.0): # EdS
					return 2.*zp1*(1.-1./np.sqrt(zp1)) * self.dH
				elif np.isclose(self.Ol0, 1.0): # dS
					return z * self.dH
				else:
					y,err = integrate.quad(_integrand, 0.0, z, epsabs=1e-8)
					return y * self.dH
			else:
				return float(1e7)

	def chi_lowz(self, z): # accepts array input for z
		return self.dH*(z - 0.5*(1.+self.q0)*z**2)

	def ztot(self, z, v=0.0):
		return (1.0 + z) * (1.0 + v/SPEEDOFLIGHT_KMS) - 1.0

	def kappa_v(self, z, v=0.0, use_lowz=False):
			dA_bar = self.chi(z, use_lowz) / (1.+z)
			dH = self.dH/self.E_Hub(z)
			return (1.0 - dH/dA_bar) * (v/SPEEDOFLIGHT_KMS)

	def dA(self, z, v=0.0, use_lowz=False):
		"""
		Computes angular diameter distance in units h^-1 kpc
		"""
		ret = self.chi(z, use_lowz) / (1.+z)
		if v == 0.0:
			ret *= 1.0
		else:
			ret *= 1.0 - self.kappa_v(z, v, use_lowz)
		return ret

	def transferFunction(self, k):
		"""
		The zero-baryon transfer function according to Eisenstein & Hu 1998.

		This fitting function is significantly simpler than the full version
		and still approximates numerical calculations from a  Boltzmann code
		to better than 10%, and almost as accurate when computing the variance
		or correlation function (see the Colossus code paper for details).
		"""
		kh = k*self.h # convert kh from hMpc^-1 to Mpc^-1

		# Equation 30
		Gamma = self.Gamma1 + self.Gamma2 / (1.0 + (0.43*kh*self.s_EH98)**4)

		# Equation 28
		q = k * self.theta2p7 * self.theta2p7 / Gamma

		# Equation 29
		C0 = 14.2 + 731.0 / (1.0 + 62.5*q)
		L0 = np.log(2.0*np.exp(1.0) + 1.8*q)
		Tk = L0 / (L0 + C0*q*q)

		return Tk

	def growthFactor(self, z): # D(a)
		return 1.0

	def growthFactor_approx(self, z):
		# The Carroll, Press & Turner (1992) approximation, eq. 29 for g(Omega)=D/a
		om_m = self.Omega_m(z)
		om_v = self.Omega_v(z)
		g = 2.5*om_m/(om_m**(4./7.)-om_v+(1.+om_m/2.)*(1.+om_v/70.))
		g0 = 2.5*self.Om0/(self.Om0**(4./7.)-self.Ol0+(1.+self.Om0/2.)*(1.+self.Ol0/70.))
		return g/g0/(1.+z) # D

	def matterPowerSpectrum(self, k, z=0.0):
		"""
		The (linear) matter power spectrum at scale k
		k has units h/Mpc so P(k) has units of [k^-3] i.e. (Mpc/h)^3
		"""
		T = self.transferFunction(k)
		D = self.growthFactor(z)
		Pk = self.As * D * D * T * T * k**self.ns
		return Pk

	def Delta2_L(self, k, z=0.0):
		"""
		Linear dimensionless matter power spectrum
		"""
		return k**3 * self.matterPowerSpectrum(k,z) / (2.*np.pi**2)

	@property
	def As(self): # scalar amplitude A_s of matter power spectrum
		if self._As is None:
			sigma80_int = self._sigmaExact()
			self._As = (self.sigma80 / sigma80_int)**2
		return self._As

	def _sigmaExact(self):
		"""
		This computes the integral of sqrt[(sigma_80)^2 / A_s].

		The infinite integral over k often causes trouble when the tophat filter is used.
		Thus we determine sensible limits and integrate over a finite k-volume.
		"""
		def _integrand(lnk):
			k = np.exp(lnk)
			x = k * 8.0
			if x < 1e-3:
				W = 1.0
			else:
				W = 3.0 / x**3 * (np.sin(x) - x * np.cos(x)) # FT of tophat filter
			T = self.transferFunction(k)
			P_unnorm = T * T * k**self.ns # equal to P(k)/A_s
			ret = P_unnorm * W**2 * k**3 # one factor of k due to the integration in log-k space
			return ret

		lnk_min, lnk_max = self._get_lnk_limits(_integrand)
		sigma2, _ = integrate.quad(_integrand, lnk_min, lnk_max, epsabs=1e-9, limit=100)
		sigma = np.sqrt(sigma2 / 2.0 / np.pi**2)
		if np.isnan(sigma):
			# raise Exception("Result is nan")
			print('sigma integral is NaN')
			print('with parameters Om0={}, sigma8={}'.format(self.Om0,self.sigma80))

		return sigma

	def _sep(self, coord_obj1, coord_obj2, use_lowz=False):
		"""
		Computes the comoving seperation between two points and
		the angles made by the two lines of sight and the connecting
		line.

		Parameters
		-------------------------------------------------------
		coord_obj1: array-like e.g. 3-tuple (z,RA,DEC)
		coord_obj2: array-like e.g. 3-tuple (z,RA,DEC)
			The angular coordinates RA and DEC are in degrees.

		Returns
		-------------------------------------------------------
		(r,theta1,theta2): 3-tuple
			r is the comoving seperation (Mpc/h)
			theta1(2) in radians is the seperation angle between the
			LOS of object 1(2) and the connecting line.

		Notes
		-------------------------------------------------------
		rhat is directed from point 1 to point 2
		"""
		deg2rad = np.pi/180
		z1, RA1, DEC1 = coord_obj1
		z2, RA2, DEC2 = coord_obj2
		alpha1 = RA1 * deg2rad
		alpha2 = RA2 * deg2rad
		delta1 = DEC1 * deg2rad
		delta2 = DEC2 * deg2rad
		nhat1 = nhat(alpha1, delta1)
		nhat2 = nhat(alpha2, delta2)
		xvec1 = self.chi(z1, use_lowz) * 1e-3 * nhat1 # since chi in kpc/h and want Mpc/h
		xvec2 = self.chi(z2, use_lowz) * 1e-3 * nhat2
		rvec = xvec2 - xvec1
		r = np.sqrt(np.dot(rvec,rvec))
		if r < 1e-14:
			theta1 = np.pi/2
			theta2 = np.pi/2
		else:
			rhat = rvec/r
			theta1 = np.arccos(np.dot(rhat,nhat1))
			theta2 = np.arccos(np.dot(rhat,nhat2))
		return r, theta1, theta2 # units radians and Mpc/h


	def xiV_perp(self, r):

		def _integrand_perp(lnk, r):
			k = np.exp(lnk)
			Pk = self.matterPowerSpectrum(k)
			x = k * r
			if x < 1e-3:
				Kperp = 1/3.
			else:
				j1 = np.sin(x)/x**2 - np.cos(x)/x
				Kperp = j1/x
			ret = k * Pk * Kperp
			ret *= (self.H0 * self.Om0**self.gamma)**2 / (2*np.pi**2)
			return ret

		if self.log_xi_perp_interpolator is not None:
			ret = 10**self.log_xi_perp_interpolator(r)
		else:
			kwargs = {'epsabs':1e-8, 'limit':100}
			lnk_min = -8
			if r > 0.0:
				lnk_max = min(3, np.log(26.6661/r)) # 8th +ve root of Kperp
			else:
				lnk_max = 3
			ret, _ = integrate.quad(_integrand_perp, lnk_min, lnk_max, args=(r,), **kwargs)
		return ret

	def xiV_para(self, r):

		def _integrand_para(lnk, r):
			k = np.exp(lnk)
			Pk = self.matterPowerSpectrum(k)
			x = k * r
			if x < 1e-3:
				Kpara = 1/3.
			else:
				j0 = np.sin(x)/x
				j1 = np.sin(x)/x**2 - np.cos(x)/x
				Kpara = j0 - 2.*j1/x
			ret = k * Pk * Kpara
			ret *= (self.H0 * self.Om0**self.gamma)**2 / (2*np.pi**2)
			return ret

		if self.xi_para_interpolator is not None:
			ret = self.xi_para_interpolator(r)
		else:
			kwargs = {'epsabs':1e-8, 'limit':100}
			lnk_min = -8
			if r > 0.0:
				lnk_max = min(3, np.log(25.0528/r)) # 8th +ve root of Kpara
			else:
				lnk_max = 3
			ret, _ = integrate.quad(_integrand_para, lnk_min, lnk_max, args=(r,), **kwargs)
		return ret

	def init_xiV_interpolation(self, rmax=400.0, Nperp=30, Npara=70, use_deriv=False):
		"""
		Notes
		-------------------------------------------------------
		To minimise number of calls to xiV_perp we note that it is a
		positive definite function and when transformed to logspace
		is close to linear which is why we use a smaller number of
		sampling points. We thus interpolate this function in logspace.

		xiV_para crosses zero so we interpolate as normal.
		"""
		self.log_xi_perp_interpolator = None
		self.xi_para_interpolator = None
		self.dlog_xi_perp_interpolator = None
		r_perp = np.linspace(0, rmax, Nperp)
		xi_perp = np.array([self.xiV_perp(r) for r in r_perp])
		if use_deriv:
			# setting s=0 interpolates all points
			self.log_xi_perp_interpolator = interpolate.UnivariateSpline(r_perp, np.log10(xi_perp), s=0, k=3)
			self.dlog_xi_perp_interpolator = self.log_xi_perp_interpolator.derivative()
		else:
			r_para = np.linspace(0, rmax, Npara)
			xi_para = np.array([self.xiV_para(r) for r in r_para])
			self.log_xi_perp_interpolator = interpolate.interp1d(r_perp, np.log10(xi_perp))
			self.xi_para_interpolator = interpolate.interp1d(r_para, xi_para)

	def xiV(self, coord_obj1, coord_obj2, use_interpolation=False, use_lowz=False, use_deriv=False):
		"""
		The velocity correlation function for two objects seperated by r in
		units Mpc/h. The two angles are the angular seperations made by the
		LOS (x2) and connecting line between each object. We assume no
		evolution in the power spectrum and hence correlation function.

		Parameters
		-------------------------------------------------------
		coord_obj1:	array-like e.g. 3-tuple (z,RA,DEC)
		coord_obj2: array-like e.g. 3-tuple (z,RA,DEC)
			The angular coordinates RA and DEC are in degrees.
		use_interpolation: bool
			If is true interpolate perp and para correlation functions
			as function of seperation r
		use_lowz: bool
			If is true evaluate distances using the low-z Taylor approximation.
		use_deriv: bool
			If is true evaluate xi_para using that xi_para = d(r * xi_perp)/dr

		Returns
		-------------------------------------------------------
		xi_V: float
			the velocity correlation in units (km/s)^2
		"""
		r, theta1, theta2 = self._sep(coord_obj1, coord_obj2, use_lowz)
		if use_interpolation:
			if self.log_xi_perp_interpolator is None:
				self.init_xiV_interpolation()

			xi_perp = 10**self.log_xi_perp_interpolator(r)
			if use_deriv:
				xi_para = xi_perp * (1. + np.log(10.)*r*self.dlog_xi_perp_interpolator(r))
			else:
				xi_para = self.xi_para_interpolator(r)
		else:
			if r < 1e-14: # points very close together so compute autocorrelation
				xi_perp = self.xiV_perp(r)
				return xi_perp # equal to xi_para which is equal to xi_v(r=0)
			else:
				xi_perp = self.xiV_perp(r)
				xi_para = self.xiV_para(r)

		ret = np.sin(theta1)*np.sin(theta2)*xi_perp + np.cos(theta1)*np.cos(theta2)*xi_para
		return ret

	@property
	def sigmav(self): # 1D velocity dispersion at z=0 in km/s
		if self._sigmav is None:
			coord = (1e-10, 1.0, 1.0)
			sigmav2 = self.xiV(coord, coord)
			self._sigmav = np.sqrt(sigmav2)
		return self._sigmav

	def xiV_correlation(self, coord_obj1, coord_obj2):
		xiV_11 = self.xiV(coord_obj1, coord_obj1)
		xiV_22 = self.xiV(coord_obj2, coord_obj2)
		xiV_12 = self.xiV(coord_obj1, coord_obj2)
		rho = xiV_12 / np.sqrt(xiV_11 * xiV_22)
		return rho

	@staticmethod
	def _get_lnk_limits(FCN_integrand, test_k_min=1e-20, test_k_max=1e20):
		"""
		The integration limits are determined by demanding that the
		integrand is some factor 1e-6 smaller than at its maximum.

		This method should be called when performing Bessel integrals.
		"""
		test_integrand_min = 1e-6
		test_lnk_min = np.log(test_k_min * 1.0001)
		test_lnk_max = np.log(test_k_max * 0.9999)

		test_lnk = np.arange(test_lnk_min, test_lnk_max, 2.0) # array of ln(k)'s
		test_k_integrand = np.zeros_like(test_lnk)
		n_test = len(test_lnk)
		for i in range(n_test):
			test_k_integrand[i] = FCN_integrand(test_lnk[i])
		integrand_max = np.max(test_k_integrand)

		min_index = 0
		while test_k_integrand[min_index] < integrand_max * test_integrand_min:
			min_index += 1
			if min_index > n_test - 2:
				raise Exception("Could not find lower integration limit")
		lnk_min = test_lnk[min_index]

		min_index -= 1
		max_index = min_index + 1
		while test_k_integrand[max_index] > integrand_max * test_integrand_min:
			max_index += 1
			if max_index == n_test:
				raise Exception("Could not find upper integration limit")
		lnk_max = test_lnk[max_index]
		return lnk_min, lnk_max

	def _get_halofit_spectral_pars(self): # Halofit
		"""
		Computes

			rknl:	wavenumber where nonlinearity begins (S03 eqn C6)
			rneff:	effective spectral index (S03 eqn C7)
			rncur:	second derivative of the power spectrum at rknl (S03 eqn C8)

		taken from Smith and Peacock halofit fortran code
		see https://www.roe.ac.uk/~jap/haloes/
		"""
		if any(p is None for p in [self.rknl, self.rneff, self.rncur]):
			xlogr1 = -2.0
			xlogr2 = 3.5
			not_converged = True
			while not_converged:
				rmid = 10**((xlogr2+xlogr1)/2.)
				sig,d1,d2 = self.wint(rmid)
				diff = sig - 1.0
				if diff > 0.001:
					xlogr1 = np.log10(rmid)
					not_converged = True
				elif diff < -0.001:
					xlogr2 = np.log10(rmid)
					not_converged = True
				else:
					self.rknl = 1./rmid
					self.rneff = -3-d1
					self.rncur = -d2
					not_converged = False
		else:
			pass

	def wint(self, r): # Halofit
		"""
		The subroutine wint, finds the effective spectral quantities
		rknl, rneff & rncur. This it does by calculating the radius of
		the Gaussian filter at which the variance is unity = rknl.
		rneff is defined as the first derivative of the variance, calculated
		at the nonlinear wavenumber and similarly the rncur is the second
		derivative at the nonlinear wavenumber.

		Taken from Smith and Peacock halofit fortran code
		see https://www.roe.ac.uk/~jap/haloes/
		"""
		nint = 3000
		t = (np.arange(nint)+0.5)/nint
		y = 1./t - 1.
		rk = y
		d2 = self.Delta2_L(rk)
		x2 = y*y*r*r
		w1 = np.exp(-x2)
		w2 = 2*x2*w1
		w3 = 4*x2*(1-x2)*w1
		fn = d2/y/t/t
		sum1 = np.sum(w1*fn)/nint
		sum2 = np.sum(w2*fn)/nint
		sum3 = np.sum(w3*fn)/nint
		sig = np.sqrt(sum1)
		d1 = -sum2/sum1
		d2 = -sum2*sum2/sum1/sum1 - sum3/sum1
		return sig, d1, d2

	def _Delta2_NL_S03(self, k, z=0.0): # Halofit Smith+ 2003 original
		self._get_halofit_spectral_pars()
		rn = self.rneff
		rncur = self.rncur
		rknl = self.rknl

		gam = 0.86485 + 0.2989*rn + 0.1631*rncur
		a = 10**(1.4861 + 1.83693*rn + 1.67618*rn*rn + 0.7940*rn*rn*rn \
			+ 0.1670756*rn*rn*rn*rn - 0.620695*rncur)
		b = 10**(0.9463 + 0.9466*rn + 0.3084*rn*rn - 0.940*rncur)
		c = 10**(-0.2807 + 0.6669*rn + 0.3214*rn*rn - 0.0793*rncur)
		xmu = 10**(-3.54419 + 0.19086*rn)
		xnu = 10**(0.95897 + 1.2857*rn)
		alpha = 1.38848 + 0.3701*rn - 0.1452*rn*rn
		beta = 0.8291 + 0.9854*rn + 0.3400*rn**2

		om_m = self.Omega_m(z)
		om_v = self.Omega_v(z)
		if abs(1-om_m) > 0.01: # omega evolution
			f1a = om_m**(-0.0732)
			f2a = om_m**(-0.1423)
			f3a = om_m**(0.0725)
			f1b = om_m**(-0.0307)
			f2b = om_m**(-0.0585)
			f3b = om_m**(0.0743)
			frac = om_v/(1.-om_m)
			f1 = frac*f1b + (1-frac)*f1a
			f2 = frac*f2b + (1-frac)*f2a
			f3 = frac*f3b + (1-frac)*f3a
		else:
			f1 = 1.0
			f2 = 1.0
			f3 = 1.0

		y = (k/rknl)

		plin = self.Delta2_L(k,z)
		ph = a*y**(f1*3) / (1+b*y**(f2)+(f3*c*y)**(3-gam))
		ph /= (1+xmu*y**(-1)+xnu*y**(-2))
		pq = plin * (1+plin)**beta/(1+plin*alpha) * np.exp(-y/4.0-y**2/8.0)

		pnl = pq + ph
		return pnl, pq, ph, plin

	def _Delta2_NL(self, k, z=0.0): # Halofit + Takahashi+ 2012
		self._get_halofit_spectral_pars()
		rn = self.rneff
		rncur = self.rncur
		rknl = self.rknl

		gam = 0.1971 - 0.0843*rn + 0.8460*rncur
		a = 10**(1.5222 + 2.8553*rn + 2.3706*rn*rn + 0.9903*rn*rn*rn \
			+ 0.2250*rn*rn*rn*rn - 0.6038*rncur)
		b = 10**(-0.5642 + 0.5864*rn + 0.5716*rn*rn - 1.5474*rncur)
		c = 10**(0.3698 + 2.0404*rn + 0.8161*rn*rn + 0.5869*rncur)
		xmu = 0.0
		xnu = 10**(5.2105 + 3.6902*rn)
		alpha = np.abs(6.0835 + 1.3373*rn - 0.1959*rn*rn - 5.5274*rncur)
		beta = 2.0379 - 0.7354*rn + 0.3157*rn**2 + 1.2490*rn**3 \
				+ 0.3980*rn**4 - 0.1682*rncur

		om_m = self.Omega_m(z)
		om_v = self.Omega_v(z)
		if np.abs(1-om_m) > 0.01: # omega evolution
			f1a = om_m**(-0.0732)
			f2a = om_m**(-0.1423)
			f3a = om_m**(0.0725)
			f1b = om_m**(-0.0307)
			f2b = om_m**(-0.0585)
			f3b = om_m**(0.0743)
			frac = om_v/(1.-om_m)
			f1 = frac*f1b + (1-frac)*f1a
			f2 = frac*f2b + (1-frac)*f2a
			f3 = frac*f3b + (1-frac)*f3a
		else:
			f1 = 1.0
			f2 = 1.0
			f3 = 1.0

		y = (k/rknl)

		plin = self.Delta2_L(k,z)
		ph = a*y**(f1*3) / (1+b*y**(f2)+(f3*c*y)**(3-gam))
		ph /= (1+xmu*y**(-1)+xnu*y**(-2))
		pq = plin * (1+plin)**beta/(1+plin*alpha) * np.exp(-y/4.0-y**2/8.0)

		pnl = pq + ph
		return pnl, pq, ph, plin

	def Delta2_NL(self, k, z=0.0): # Halofit
		pnl,_,_,_ = self._Delta2_NL(k,z)
		return pnl

	def matterPowerSpectrum_NL(self, k, z=0.0):
		"""
		Nonlinear matter power spectrum
		k has units h/Mpc so P(k) has units of [k^-3] i.e. (Mpc/h)^3
		"""
		return self.Delta2_NL(k,z) * (2.*np.pi**2) / k**3



