#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
from util import read_signal,get_frame,add_signal,write_signal,compute_avgspectrum,separate_channels,uniting_channles
from voice_enhancement import SupectralSubtruction,MMSE_STSA,JointMap,MMSE_LogSTSA
import optparse

WINSIZE=2048
#songfile='denoise_derev_in1.wav'
#outfile='denoise_derev_in1_lmmse.wav'


def noise_reduction(signal,winsize,window):
    out=sp.zeros(len(signal),sp.float32)
    n_amp = compute_avgspectrum(signal[0:winsize*20],winsize,window)
    for no in xrange(nf):
        s = get_frame(signal, winsize, no)
        add_signal(out, ss.compute_by_noise_amp(s,n_amp), winsize, no)
    return out

if __name__=="__main__":
    parser = optparse.OptionParser(usage="%prog [-m METHOD] [-w WINSIZE] INPUTFILE OUTPUTFILE\n method 0 : SupectralSubtruction\n        1 : MMSE_STSA\n        2 : MMSE_LogSTSA\n        3 : JointMap")

    parser.add_option("-w", type="int", dest="winsize", default=WINSIZE)
    parser.add_option("-m", type="int", dest="method", default=0)

    (options, args) = parser.parse_args()

    if len(args)!=2:
        parser.print_help()
        exit(2)
    
    signal, params = read_signal(args[0],options.winsize)
    nf = len(signal)/(options.winsize/2) - 1

    window = sp.hanning(options.winsize)
    if options.method==0:
        ss = SupectralSubtruction(options.winsize,window)
    elif options.method==1:
        ss = MMSE_STSA(options.winsize,window)
    elif options.method==2:
        ss = MMSE_LogSTSA(options.winsize,window)
    elif options.method==3:
        ss = JointMap(options.winsize,window)

    if params[0]==1:
        write_signal(args[1], params, noise_reduction(signal,options.winsize,window))
    elif params[0]==2:
        l,r = separate_channels(signal)
        write_signal(args[1], params, uniting_channles(noise_reduction(l,options.winsize,window),noise_reduction(r,options.winsize,window)))
