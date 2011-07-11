#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp


def lpc(signal,order,winsize,window):
    """
    Calucurate LPC by PARCOR analysis
    """
    signal = signal*window
    a = sp.zeros(order+1,sp.float32)
    k = sp.zeros(order+1,sp.float32)
    r = sp.signal.correlate(signal)[signal.size-1:signal.size+order]
    a[0]=1.0
    w=r[1]
    u=r[0]
    for m in xrange(1,order+1):
        k[m] = w/u
        u = u*(1.0-k[m]**2.0)
        for i in xrange(1,m+1):
            a[i] = a[i]-k[m]*a[m-i]
        w=0.0
        for i in xrange(m+1):
            w+=a[i]*r[m+1-i]
    return a,k
