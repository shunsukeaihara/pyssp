#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
from util import get_frame

def itakura_saito_distotion_measure(s,s_hat):
    size = min(len(s),len(s_hat))
    s = s[0:size]
    s_hat = s[0:size]
    return sp.sum(s - s_hat + sp.exp(s_hat)/sp.exp(s)-1.0)/size


def segmental_itakura_saito_distotion_measure(s,s_hat,winsize):
    size = min(len(s),len(s_hat))
    nf = size/(winsize/2) - 1
    ret=[]
    for no in xrange(nf):
        s_i = get_frame(s, winsize, no)
        s_hat_i = get_frame(s_hat, winsize, no)
        ret.append(itakura_saito_distotion_measure(s_i,s_hat_i))
    return ret
