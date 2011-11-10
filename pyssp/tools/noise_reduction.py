#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
import sys
import wave
import tempfile
from pyssp.util import read_signal,get_frame,add_signal,separate_channels,uniting_channles,compute_avgpowerspectrum
from pyssp.voice_enhancement import SupectralSubtruction,MMSE_STSA,JointMap,MMSE_LogSTSA
import optparse


def noise_reduction(signal,params,winsize,window,ss,ntime):
    out=sp.zeros(len(signal),sp.float32)
    n_pow = compute_avgpowerspectrum(signal[0:winsize*int(params[2] /float(winsize)/(1000.0/ntime))],winsize,window)#maybe 300ms
    nf = len(signal)/(winsize/2) - 1
    for no in xrange(nf):
        s = get_frame(signal, winsize, no)
        add_signal(out, ss.compute_by_noise_pow(s,n_pow), winsize, no)
    return out


def write(param,signal):
    st = tempfile.TemporaryFile()
    wf=wave.open(st,'wb')
    wf.setparams(params)
    s=sp.int16(signal*32767.0).tostring()
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
        signal=sp.zeros(siglen, sp.float32)
        signal[0:len(str)/2] = sp.float32(sp.fromstring(str,sp.int16))/32767.0
        return signal,params
    else:
        return read_signal(fname,winsize)


if __name__=="__main__":
    parser = optparse.OptionParser(usage="%prog [-m METHOD] [-w WINSIZE] [- s NOISETIME(ms)] INPUTFILE\n method 0 : SupectralSubtruction\n        1 : MMSE_STSA\n        2 : MMSE_LogSTSA\n        3 : JointMap\n if INPUTFILE is \"-\", read wave data from stdin")

    parser.add_option("-w", type="int", dest="winsize", default=512)
    parser.add_option("-m", type="int", dest="method", default=0)
    parser.add_option("-t", type="int", dest="ntime", default=300)
    parser.add_option("-s", type="float", dest="sscoefficient", default=4.0)
    parser.add_option("-r", type="float", dest="ratio", default=1.0)
    parser.add_option("-c", type="float", dest="constant", default=0.001)
    parser.add_option("-a", type="float", dest="alpha", default=0.99)
    (options, args) = parser.parse_args()

    if len(args)!=1:
        parser.print_help()
        exit(2)

    fname = args[0]
    signal, params = read(fname,options.winsize)

    window = sp.hanning(options.winsize)
    import os.path
    
    root,ext = os.path.splitext(args[0])
    if options.method==0:
        ss = SupectralSubtruction(options.winsize,window,ratio=options.ratio,coefficient=options.sscoefficient)
        outfname = "%s_ss%s" % (root,ext)
    elif options.method==1:
        ss = MMSE_STSA(options.winsize,window,ratio=options.ratio,constant=options.constant,alpha=options.alpha)
        outfname = "%s_mmse%s" % (root,ext)
    elif options.method==2:
        ss = MMSE_LogSTSA(options.winsize,window,ratio=options.ratio,alpha=options.alpha,constant=options.constant)
        outfname = "%s_lmmse%s" % (root,ext)
    elif options.method==3:
        ss = JointMap(options.winsize,window,alpha=options.alpha,ratio=options.ratio,constant=options.constant)
        outfname = "%s_jm%s" % (root,ext)

    if params[0]==1:
        write(params, noise_reduction(signal,params,options.winsize,window,ss,options.ntime))
    elif params[0]==2:
        l,r = separate_channels(signal)
        write(params, uniting_channles(noise_reduction(l,params,options.winsize,window,ss,options.ntime),noise_reduction(r,params,options.winsize,window,ss,options.ntime)))
