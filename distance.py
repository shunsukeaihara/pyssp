#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
from util import get_frame

def itakura_saito_distotion_measure(s,shat):
    size = min(len(s),len(shat))
    s = s[0:size]
    shat = shat[0:size]
    return sp.sum(s - shat + sp.exp(shat)/sp.exp(s)-1.0)/float(size)


def segmental_itakura_saito_distotion_measure(s,shat,winsize):
    size = min(len(s),len(shat))
    nf = size/(winsize/2) - 1
    ret=[]
    for no in xrange(nf):
        s_i = get_frame(s, winsize, no)
        shat_i = get_frame(shat, winsize, no)
        ret.append(itakura_saito_distotion_measure(s_i,shat_i))
    return ret


def itakura_saito_spectrum_distance(s,shat,winfunc):
    size = min(len(s),len(shat))
    window = winfunc(size)
    s = s[0:size]
    shat = shat[0:size]
    s_amp = sp.absolute(sp.fft(s*window))
    shat_amp = sp.absolute(sp.fft(shat*window))
    return sp.mean(s_amp / shat_amp - sp.log10(s_amp / shat_amp) - 1.0)


def segmental_itakura_saito_spectrum_distance(s,shat,winsize,winfunc):
    size = min(len(s),len(shat))
    nf = size/(winsize/2) - 1
    ret=[]
    for no in xrange(nf):
        s_i = get_frame(s, winsize, no)
        shat_i = get_frame(shat, winsize, no)
        ret.append(itakura_saito_spectrum_distance(s_i,shat_i))
    return ret

if __name__=="__main__":
    pass
