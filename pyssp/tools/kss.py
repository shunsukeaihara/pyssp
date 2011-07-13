#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
import math
from pyssp.util import read_signal, get_frame,separate_channels,add_signal,uniting_channles
from pyssp.voice_enhancement import SupectralSubtruction,MMSE_STSA,JointMap,MMSE_LogSTSA
import optparse
import tempfile
import wave

class KaraokeFileLoader():
    def __init__(self,winsize):
        self._winsize = winsize

    def load_file(self,songfile,karaokefile):
        ssignal, params = read_signal(songfile,self._winsize)
        ksignal, params = read_signal(karaokefile,self._winsize)
        sindex,kindex = self._alignment(ssignal,ksignal)
        s,k = self._reshape_signal(sindex,kindex,ssignal,ksignal)
        return s,k,params

    def _reshape_signal(self,sindex,kindex,ssignal,ksignal):
        def reshape(signal,siglen,winsize):
            length =(siglen/winsize+1)*winsize
            ret=sp.zeros(length, sp.float32)
            ret[0:siglen] = signal
            return ret
        slen = len(ssignal)-sindex
        klen = len(ksignal)-kindex
        length = 0
        if slen>klen:
            length = klen
        else:
            length = slen
        ssignal=reshape(ssignal[sindex:sindex+length],length,self._winsize)
        ksignal=reshape(ksignal[kindex:kindex+length],length,self._winsize)
        return ssignal,ksignal
        
    def _alignment(self,ssignal,ksignal):
        starta = 0
        for i in range(len(ssignal))[0::2]:
            if ssignal[i]<-100/32767.0 or ssignal[i]>100/32767.0:
                starta = i
                break
        startb=0
        for i in range(len(ksignal))[0::2]:
            if ksignal[i]<-100/32767.0 or ksignal[i]>100/32767.0:
                startb = i
                break
        start=starta-100
        base = ssignal[start:start+5000]
        small=1000000
        index=0
        for i in range(startb-1000,startb-1000+10000)[0::2]:
            signal = ksignal[i:i+5000]
            score =  math.sqrt(sp.sum(sp.square(sp.array(list(base-signal),sp.float32))))
            if score<small:
                index=i
                small=score
        return  start,index
        #return 0,0

def subtruction(ssignal,ksignal,window,winsize,method):
    nf = len(ssignal)/(winsize/2) - 1
    out=sp.zeros(len(ssignal),sp.float32)
    for no in xrange(nf):
        s = get_frame(ssignal, winsize, no)
        k = get_frame(ksignal, winsize, no)
        add_signal(out, method.compute(s,k), winsize, no)
    return out

def fin(size,signal):
    fil = sp.zeros(size,sp.float32)
    for i in xrange(size):
        ratio=sp.log10((i+1)/float(size)*10+1.0)
        if ratio>1.0:
            ratio=1.0
        fil[i] = ratio
    return fil*signal

def fout(size,signal):
    fil = sp.zeros(size,sp.float32)
    for i in xrange(size):
        ratio = sp.log10((size-i)/float(size)*10+1.0)
        if ratio>1.0:
            ratio = 1.0
        fil[i] = ratio
    return fil*signal

def vad(vas,signal,winsize,window):
    out=sp.zeros(len(signal),sp.float32)
    for va in vas:
        for i in range(va[0],va[1]):
            add_signal(out,get_frame(signal, winsize, i)*window,winsize,i)
    for va in vas:
        out[(va[0])*winsize/2:(va[0]+4)*winsize/2] = fin(winsize*2,out[(va[0])*winsize/2:(va[0]+4)*winsize/2])
        out[(va[1]-4)*winsize/2:(va[1])*winsize/2] = fout(winsize*2,out[(va[1]-4)*winsize/2:(va[1])*winsize/2])
    return out

def write(param,signal):
    st = tempfile.TemporaryFile()
    wf=wave.open(st,'wb')
    wf.setparams(params)
    s=sp.int16(signal*32767.0).tostring()
    wf.writeframes(s)
    st.seek(0)
    print st.read()


if __name__ == "__main__":
    parser = optparse.OptionParser(usage="%prog [-m METHOD] [-w WINSIZE] SONGFILE KARAOKEFILE\n method 0 : SupectralSubtruction\n        1 : MMSE_STSA\n        2 : MMSE_LogSTSA\n        3 : JointMap\n if INPUTFILE is \"-\", read wave data from stdin")

    parser.add_option("-w", type="int", dest="winsize", default=1024)
    parser.add_option("-m", type="int", dest="method", default=0)

    (options, args) = parser.parse_args()

    if len(args)!=2:
        parser.print_help()
        exit(2)

    
    kl = KaraokeFileLoader(options.winsize*2)

    ssignal,ksignal,params = kl.load_file(args[0],args[1])
    ssignal_l,ssignal_r = separate_channels(ssignal)
    ksignal_l,ksignal_r = separate_channels(ksignal)

    window = sp.hanning(options.winsize)

    if options.method==0:
        method = SupectralSubtruction(options.winsize,window)
    elif options.method==1:
        method = MMSE_STSA(options.winsize,window)
    elif options.method==2:
        method = MMSE_LogSTSA(options.winsize,window,alpha=0.99)
    elif options.method==3:
        method = JointMap(options.winsize,window,alpha=0.99)

    sig_out_l = subtruction(ssignal_l,ksignal_l,window,options.winsize,method)
    sig_out_r = subtruction(ssignal_r,ksignal_r,window,options.winsize,method)

    sig_out_l[sp.isnan(sig_out_l)+sp.isinf(sig_out_l)]=0.0
    sig_out_r[sp.isnan(sig_out_r)+sp.isinf(sig_out_r)]=0.0


    result = uniting_channles(sig_out_l, sig_out_r)
    write(params, result)
    
    
