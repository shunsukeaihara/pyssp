#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
import scipy.special as spc

class SupectralSubtruction():
    def __init__(self,winsize,window,ratio=2.0):
        self._window=window
        self._ratio=ratio
            
    def compute(self,signal,noise):
        s_spec = sp.fft(signal*self._window)
        s_amp = sp.absolute(s_spec)**2.0
        s_phase = sp.angle(s_spec)
        n_spec = sp.fft(noise*self._window)
        n_amp = sp.absolute(n_spec)**2.0
        s_amp = s_amp - n_amp*self._ratio
        s_amp = sp.maximum(s_amp,0)
        s_amp = sp.sqrt(s_amp)
        s_spec = s_amp * sp.exp(s_phase*1j)
        return sp.real(sp.ifft(s_spec))

class SpectrumReconstruction(object):
    def __init__(self,winsize,window,alpha=0.98):
        self._window=window
        self._G = sp.zeros(winsize,sp.float32)
        self._prevGamma = sp.zeros(winsize,sp.float32)
        self._alpha = alpha

    def compute(self,signal,noise):
        return signal

    def _calc_aposteriori_snr(self,s_amp,n_amp):
        return s_amp**2.0/n_amp**2.0

    def _calc_apriori_snr(self,gamma):
        return self._alpha*self._G**2 * self._prevGamma + (1.0-self._alpha)*sp.maximum(gamma-1.0, 0)#a priori s/n ratio

class MMSE_STSA(SpectrumReconstruction):
    def __init__(self,winsize,window,alpha=0.99):
        self._gamma15=spc.gamma(1.5)
        super(self.__class__,self).__init__(winsize,window,alpha)
            
    def compute(self,signal,noise):
        s_spec = sp.fft(signal*self._window)
        s_amp = sp.absolute(s_spec)
        s_phase = sp.angle(s_spec)
        n_spec = sp.fft(noise*self._window)
        n_amp = sp.absolute(n_spec)
        gamma = self._calc_aposteriori_snr(s_amp,n_amp)
        xi = self._calc_apriori_snr(gamma)
        self._prevGamma = gamma
        nu = gamma * xi / (1+xi)
        self._G = (self._gamma15*sp.sqrt(nu)/gamma)*sp.exp(-nu/2)*((1+nu)*spc.i0(nu/2)+nu*spc.i1(nu/2))
        idx = sp.isnan(self._G) + sp.isinf(self._G)
        #self._G[idx] = xi[idx] / ( xi[idx] + 1)
        self._G[idx] = 0.0
        amp = self._G * s_amp
        amp = sp.maximum(amp,0)
        spec = amp * sp.exp(s_phase*1j)
        return sp.real(sp.ifft(spec))

class JointMap(SpectrumReconstruction):
    def __init__(self,winsize,window,alpha=0.99,mu=1.74,tau=0.126):
        self._mu = mu
        self._tau = tau
        super(self.__class__,self).__init__(winsize,window,alpha)
            
    def compute(self,signal,noise):
        s_spec = sp.fft(signal*self._window)
        s_amp = sp.absolute(s_spec)
        s_phase = sp.angle(s_spec)
        n_spec = sp.fft(noise*self._window)
        n_amp = sp.absolute(n_spec)
        gamma = self._calc_aposteriori_snr(s_amp,n_amp)
        xi = self._calc_apriori_snr(gamma)
        self._prevGamma = gamma
        u = 0.5 - self._mu/(4.0*sp.sqrt(gamma*xi))
        self._G = u + sp.sqrt(u**2.0 + self._tau/(gamma*2.0))
        idx = sp.isnan(self._G) + sp.isinf(self._G)
        #self._G[idx] = xi[idx] / ( xi[idx] + 1)
        self._G[idx] = 0.0
        amp = self._G * s_amp
        spec = amp * sp.exp(s_phase*1j)
        return sp.real(sp.ifft(spec))
