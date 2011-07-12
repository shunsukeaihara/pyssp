#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
import sys
import os
from pyssp.util import get_frame,add_signal,read_signal,write_signal,separate_channels,uniting_channles
from pyssp.vad.ltsd import LTSD

def vad(vas,signal,winsize,window):
    out=sp.zeros(len(signal),sp.float32)
    for va in vas:
        for i in range(va[0],va[1]+2):
            add_signal(out,get_frame(signal, winsize, i)*window,winsize,i)
    return out

if __name__ == "__main__":
    """
    python vad.py WINSIZE THREATHOLD FILENAME
    """
    windowsize = int(sys.argv[1])
    signal, params = read_signal(sys.argv[3],windowsize)
    root,ext = os.path.splitext(sys.argv[3])
    outfname = "%s_vaded%s" % (root,ext)
    window = sp.hanning(windowsize)

    if params[0]==1:
        ltsd = LTSD(windowsize,window,5,lambda0=int(sys.argv[2]))
        res,ltsds =  ltsd.compute_without_noise(signal)
        write_signal(outfname,params,vad(res,signal,windowsize,window))
    elif params[0]==2:
        l,r = separate_channels(signal)
        ltsd_l = LTSD(windowsize,window,5,lambda0=int(sys.argv[2]))
        ltsd_r = LTSD(windowsize,window,5,lambda0=int(sys.argv[2]))
        out = uniting_channles(vad(ltsd_l.compute_without_noise(l)[0],l,windowsize,window),
                               vad(ltsd_r.compute_without_noise(r)[0],r,windowsize,window))
        write_signal(outfname,params,out)
