#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
import sys
import wave
import tempfile
from pyssp.util import get_frame,add_signal,read_signal
from pyssp.vad.ltsd import LTSD
import optparse
from scikits.learn.hmm import GaussianHMM
import pickle

WINSIZE = 1024

def vad(vas,signal,winsize,window):
    out=sp.zeros(len(signal),sp.float32)
    for va in vas:
        for i in range(va[0],va[1]+5):
            add_signal(out,get_frame(signal, winsize, i)*window,winsize,i)
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
        signal[0:len(str)/2] =  sp.float32(sp.fromstring(str,sp.int16))/32767.0
        return signal,params
    else:
        return read_signal(fname,winsize)


def get_model(path,npath):
    f = open(path)
    mhmm = pickle.load(f)
    f.close()
    f = open(npath)
    nhmm = pickle.load(f)
    f.close()
    return mhmm,nhmm

def hmm_filter(mhmm,nhmm,signal,vas,winsize,window):
    ret = []
    for va in vas:
        ls = []
        for i in range(va[0],va[1]+2):
            s = get_frame(signal,winsize,i)
            s_spec = sp.fft(s*window)
            ls.append(sp.absolute(s_spec))
        print mhmm.score(ls)
        print nhmm.score(ls)
        if mhmm.score(ls) > nhmm.score(ls):
            ret.append(va)
    return ret
    
if __name__ == "__main__":
    """
    python vad.py -w WINSIZE -t THREATHOLD FILENAME
    """
    parser = optparse.OptionParser(usage="%python vad INPUTFILE \n if INPUTFILE is \"-\", read wave data from stdin")
    parser.add_option("-t", type="int", dest="th", default=10)
    (options, args) = parser.parse_args()
    windowsize = 1024

    fname = args[0]
    signal, params = read(fname,windowsize)
    window = sp.hanning(windowsize)

    mhmm,nhmm = get_model("hmm.model","noisyhmm.model")

    if params[0]==1:
        ltsd = LTSD(windowsize,window,6,lambda0=options.th)
        res,ltsds =  ltsd.compute_with_noise(signal,signal[0:windowsize*int(params[2] /float(windowsize)/3.0)])#maybe 300ms
        print res
        res = hmm_filter(mhmm,nhmm,signal,res,windowsize,window)
        print res
        #write(params,vad(res,signal,windowsize,window))
    elif params[0]==2:
        write(params,signal)
