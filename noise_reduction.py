#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
from util import read_signal, get_frame,separate_channels,add_signal,uniting_channles,write_signal
from voice_enhancement import SupectralSubtruction,MMSE_STSA,JointMap,MMSE_LogSTSA

WINSIZE=2048
songfile='denoise_derev_in1.wav'
outfile='denoise_derev_in1_lmmse.wav'


def compute_noise_avgspectrum(nsignal,winsize,window):
    windownum = len(nsignal)/(winsize/2) - 1
    avgamp = sp.zeros(winsize)
    for l in xrange(windownum):
        avgamp += sp.absolute(sp.fft(get_frame(nsignal, winsize,l) * window))
    return avgamp/float(windownum)


if __name__=="__main__":
    signal, params = read_signal(songfile,WINSIZE)
    nf = len(signal)/(WINSIZE/2) - 1
    out=sp.zeros(len(signal),sp.float32)
    window = sp.hanning(WINSIZE)
    #ss = JointMap(WINSIZE,window)
    ss = MMSE_LogSTSA(WINSIZE,window)
    n_amp = compute_noise_avgspectrum(signal[0:WINSIZE*20],WINSIZE,window)

    for no in xrange(nf):
        s = get_frame(signal, WINSIZE, no)
        add_signal(out, ss.compute_by_noise_amp(s,n_amp), WINSIZE, no)

    write_signal(outfile, params, out)
    
