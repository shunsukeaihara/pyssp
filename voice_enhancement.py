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
        SYr = sp.maximum(SYr,0)#ゼロ以下を切り捨て
        SYr = sp.sqrt(SYr)
        SY = SYr * sp.exp(SYp*1j)
        return sp.real(sp.ifft(SY))


class MMSE_STSA():
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
        nlambda = sp.absolute(NY)
        gamma = SYr**2.0/nlambda**2.0 #a-posteriori s/n ratio
        xi = self._alpha*self._G**2 * self._prevGamma + (1.0-self._alpha)*sp.maximum(gamma-1.0, 0)
        self._prevGamma = gamma
        nu = gamma * xi / (1+xi)
        self._G = (self._gamma15*sp.sqrt(nu)/gamma)*sp.exp(-nu/2)*((1+nu)*spc.i0(nu/2)+nu*spc.i1(nu/2))
        idx = sp.isnan(self._G) + sp.isinf(self._G)
        self._G[idx] = xi[idx] / ( xi[idx] + 1)
        Yr = self._G * SYr
        Y = Yr * sp.exp(SYp*1j)
        return sp.real(sp.ifft(Y))

class JointMap():
    def __init__(self,winsize,window,alpha=0.98):
        self._window=window
        self._G = sp.zeros(winsize,sp.float32)
        self._prevGamma = sp.ones(winsize,sp.float32)
        self._alpha = alpha
        self._gamma15=spc.gamma(1.5)
            
    def compute(self,signal,noise):
        SY = sp.fft(signal*self._window)
        SYr = sp.absolute(SY)
        SYp = sp.angle(SY)
        NY = sp.fft(noise*self._window)
        nlambda = sp.absolute(NY)
        gamma = SYr**2.0/nlambda**2.0 #a-posteriori s/n ratio
        xi = self._alpha*self._G**2 * self._prevGamma + (1.0-self._alpha)*sp.maximum(gamma-1.0, 0)
        self._prevGamma = gamma
        nu = gamma * xi / (1+xi)
        self._G = (self._gamma15*sp.sqrt(nu)/gamma)*sp.exp(-nu/2)*((1+nu)*spc.i0(nu/2)+nu*spc.i1(nu/2))
        idx = sp.isnan(self._G) + sp.isinf(self._G)
        self._G[idx] = xi[idx] / ( xi[idx] + 1)
        Yr = self._G * SYr
        Y = Yr * sp.exp(SYp*1j)
        return sp.real(sp.ifft(Y))
