#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
import scipy.special as spc

class SupectralSubtruction():
    def __init__(self,winsize,window,ratio=2.0):
        self._window=window
        self._ratio=ratio
            
    def compute(self,signal,noise):
        SY = sp.fft(signal*self._window)
        SYr = sp.absolute(SY)**2.0
        SYp = sp.angle(SY)
        NY = sp.fft(noise*self._window)
        NYr = sp.absolute(NY)**2.0
        SYr = SYr - NYr*self._ratio
        SYr = sp.maximum(SYr,0)
        SYr = sp.sqrt(SYr)
        SY = SYr * sp.exp(SYp*1j)
        return sp.real(sp.ifft(SY))

class SpectrumReconstruction():
    def __init__(self,winsize,window,alpha=0.98):
        self._window=window
        self._G = sp.zeros(winsize,sp.float32)
        self._prevGamma = sp.zeros(winsize,sp.float32)
        self._alpha = alpha

    def compute(self,signal,noise):
        return signal

    def _calc_aposteriori_snr(self,syr,nyr):
        return syr**2.0/nyr**2.0

    def _calc_apriori_snr(self,gamma):
        return self._alpha*self._G**2 * self._prevGamma + (1.0-self._alpha)*sp.maximum(gamma-1.0, 0)#a priori s/n ratio

class MMSE_STSA(SpectrumReconstruction):
    def __init__(self,winsize,window,alpha=0.98):
        self._window=window
        self._G = sp.zeros(winsize,sp.float32)
        self._prevGamma = sp.zeros(winsize,sp.float32)
        self._alpha = alpha
        self._gamma15=spc.gamma(1.5)
            
    def compute(self,signal,noise):
        SY = sp.fft(signal*self._window)
        SYr = sp.absolute(SY)
        SYp = sp.angle(SY)
        NY = sp.fft(noise*self._window)
        NYr = sp.absolute(NY)
        gamma = self._calc_aposteriori_snr(SYr,NYr)
        xi = self._calc_apriori_snr(gamma)
        self._prevGamma = gamma
        nu = gamma * xi / (1+xi)
        self._G = (self._gamma15*sp.sqrt(nu)/gamma)*sp.exp(-nu/2)*((1+nu)*spc.i0(nu/2)+nu*spc.i1(nu/2))
        idx = sp.isnan(self._G) + sp.isinf(self._G)
        #self._G[idx] = xi[idx] / ( xi[idx] + 1)
        self._G[idx] = 0.0
        Yr = self._G * SYr
        Yr = sp.maximum(Yr,0)
        Y = Yr * sp.exp(SYp*1j)
        return sp.real(sp.ifft(Y))

class JointMap(SpectrumReconstruction):
    def __init__(self,winsize,window,alpha=0.99,mu=1.74,tau=0.126):
        self._window=window
        self._G = sp.zeros(winsize,sp.float32)
        self._prevGamma = sp.zeros(winsize,sp.float32)
        self._alpha = alpha
        self._mu = mu
        self._tau = tau
            
    def compute(self,signal,noise):
        SY = sp.fft(signal*self._window)
        SYr = sp.absolute(SY)
        SYp = sp.angle(SY)
        NY = sp.fft(noise*self._window)
        NYr = sp.absolute(NY)
        gamma = self._calc_aposteriori_snr(SYr,NYr)
        xi = self._calc_apriori_snr(gamma)
        self._prevGamma = gamma
        u = 0.5 - self._mu/(4.0*sp.sqrt(gamma*xi))
        self._G = u + sp.sqrt(u**2.0 + self._tau/(gamma*2.0))
        idx = sp.isnan(self._G) + sp.isinf(self._G)
        #self._G[idx] = xi[idx] / ( xi[idx] + 1)
        self._G[idx] = 0.0
        Yr = self._G * SYr
        Y = Yr * sp.exp(SYp*1j)
        return sp.real(sp.ifft(Y))
