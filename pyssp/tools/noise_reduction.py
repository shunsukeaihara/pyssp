#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
import sys
import wave
import tempfile
from pyssp.util import read_signal,get_frame,add_signal,separate_channels,uniting_channles,compute_avgpowerspectrum
from pyssp.voice_enhancement import SupectralSubtruction,MMSE_STSA,JointMap,MMSE_LogSTSA
import optparse

WINSIZE=1024

def noise_reduction(signal,winsize,window,ss):
    out=sp.zeros(len(signal),sp.float32)
    n_pow = compute_avgpowerspectrum(signal[0:winsize*5],winsize,window)
    for no in xrange(nf):
        s = get_frame(signal, winsize, no)
        add_signal(out, ss.compute_by_noise_pow(s,n_pow), winsize, no)
    return out


def write(param,signal):
    st = tempfile.TemporaryFile()
    wf=wave.open(st,'wb')
    wf.setparams(params)
    s=sp.int16(signal).tostring()
    wf.writeframes(s)
    st.seek(0)
    print st.read()

def read(fname,winsize):
    if fname =="-":
        wf=wave.open(sys.stdin,'rb')
        n=wf.getnframes()
        str=wf.readframes(n)
        params = ((wf.getnchannels(), wf.getsampwidth(),
                   wf.getframerate(), wf.getnframes(),
                   wf.getcomptype(), wf.getcompname()))
        siglen=((int )(len(str)/2/winsize) + 1) * winsize
        signal=sp.zeros(siglen, sp.int16)
        signal[0:len(str)/2] = sp.fromstring(str,sp.int16)
        return signal,params
    else:
        return read_signal(fname,winsize)


if __name__=="__main__":
    parser = optparse.OptionParser(usage="%prog [-m METHOD] [-w WINSIZE] INPUTFILE\n method 0 : SupectralSubtruction\n        1 : MMSE_STSA\n        2 : MMSE_LogSTSA\n        3 : JointMap\n if INPUTFILE is \"-\", read wave data from stdin")

    parser.add_option("-w", type="int", dest="winsize", default=WINSIZE)
    parser.add_option("-m", type="int", dest="method", default=0)

    (options, args) = parser.parse_args()

    if len(args)!=1:
        parser.print_help()
        exit(2)

    fname = args[0]
    signal, params = read(fname,options.winsize)

    nf = len(signal)/(options.winsize/2) - 1
    window = sp.hanning(options.winsize)
    import os.path
    
    root,ext = os.path.splitext(args[0])
    if options.method==0:
        ss = SupectralSubtruction(options.winsize,window)
        outfname = "%s_ss%s" % (root,ext)
    elif options.method==1:
        ss = MMSE_STSA(options.winsize,window)
        outfname = "%s_mmse%s" % (root,ext)
    elif options.method==2:
        ss = MMSE_LogSTSA(options.winsize,window,alpha=0.99)
        outfname = "%s_lmmse%s" % (root,ext)
    elif options.method==3:
        ss = JointMap(options.winsize,window,alpha=0.99)
        outfname = "%s_jm%s" % (root,ext)

    if params[0]==1:
        write(params, noise_reduction(signal,options.winsize,window,ss))
    elif params[0]==2:
        l,r = separate_channels(signal)
        write(params, uniting_channles(noise_reduction(l,options.winsize,window,ss),noise_reduction(r,options.winsize,window,ss)))
