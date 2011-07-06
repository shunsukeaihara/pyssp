#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
import math
from util import read_signal, get_frame,separate_channels,add_signal,uniting_channles,write_signal
from vad.ltsd import LTSD
from voice_enhancement import SupectralSubtruction,MMSE_STSA,JointMap,MMSE_LogSTSA

WINSIZE=4096
VADOFFSET = 1
songfile='maniac.wav'
karaokefile="maniac_offv.wav"
outfile="maniac_lmmse.wav"

class KaraokeFileLoader():
    def __init__(self,winsize):
        self._winsize = winsize

    def load_file(self,songfilepath,karaokefile):
        ssignal, params = read_signal(songfile,self._winsize)
        ksignal, params = read_signal(karaokefile,self._winsize)
        sindex,kindex = self._alignment(ssignal,ksignal)
        s,k = self._reshape_signal(sindex,kindex,ssignal,ksignal)
        return s,k,params

    def _reshape_signal(self,sindex,kindex,ssignal,ksignal):
        def reshape(signal,siglen,winsize):
            length =(siglen/winsize+1)*winsize
            ret=sp.zeros(length, sp.int16)
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
            if ssignal[i]<-100 or ssignal[i]>100:
                starta = i
                break
        startb=0
        for i in range(len(ksignal))[0::2]:
            if ksignal[i]<-100 or ksignal[i]>100:
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


def subtruction(ssignal,ksignal,window,winsize):
    nf = len(ssignal)/(winsize/2) - 1
    out=sp.zeros(len(ssignal),sp.float32)
    #ss = SupectralSubtruction(winsize,window)
    #ss = MMSE_STSA(winsize,window)
    ss = MMSE_LogSTSA(winsize,window)
    #ss = JointMap(winsize,window)
    for no in xrange(nf):
        s = get_frame(ssignal, winsize, no)
        k = get_frame(ksignal, winsize, no)
        add_signal(out, ss.compute(s,k), winsize, no)
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


if __name__ == "__main__":
    kl = KaraokeFileLoader(WINSIZE*2)

    ssignal,ksignal,params = kl.load_file(songfile,karaokefile)
    ssignal_l,ssignal_r = separate_channels(ssignal)
    ksignal_l,ksignal_r = separate_channels(ksignal)

    print "Alignment is done"
    window = sp.hanning(WINSIZE)

    sig_out_l = subtruction(ssignal_l,ksignal_l,window,WINSIZE)
    sig_out_r = subtruction(ssignal_r,ksignal_r,window,WINSIZE)
    print "Spectral Subtraction is Done"

    sig_out_l[sp.isnan(sig_out_l)+sp.isinf(sig_out_l)]=0.0
    sig_out_r[sp.isnan(sig_out_r)+sp.isinf(sig_out_r)]=0.0

    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = fig.add_subplot(321)
    #Pxx,freqs, bins, im = ax.specgram(ssignal_l[0:200*WINSIZE],
    #                               NFFT=WINSIZE, Fs=44100,
    #                               noverlap=WINSIZE/2, window=window)
    #ax2 = fig.add_subplot(323)
    #Pxx,freqs, bins, im = ax2.specgram(ksignal_l[0:200*WINSIZE],
    #                               NFFT=WINSIZE, Fs=44100,
    #                               noverlap=WINSIZE/2, window=window)
    #ax3 = fig.add_subplot(325)
    #Pxx,freqs, bins, im = ax3.specgram(sig_out_l[0:200*WINSIZE],
    #                               NFFT=WINSIZE, Fs=44100,
    #                               noverlap=WINSIZE/2, window=window)
    #ax4 = fig.add_subplot(322)
    #Pxx,freqs, bins, im = ax4.specgram(ssignal_r[0:200*WINSIZE],
    #                               NFFT=WINSIZE, Fs=44100,
    #                               noverlap=WINSIZE/2, window=window)
    #ax5 = fig.add_subplot(324)
    #Pxx,freqs, bins, im = ax5.specgram(ksignal_r[0:200*WINSIZE],
    #                               NFFT=WINSIZE, Fs=44100,
    #                               noverlap=WINSIZE/2, window=window)
    #ax6 = fig.add_subplot(326)
    #Pxx,freqs, bins, im = ax6.specgram(sig_out_r[0:200*WINSIZE],
    #                               NFFT=WINSIZE, Fs=44100,
    #                               noverlap=WINSIZE/2, window=window)
    #plt.show()

    """
    ltsd = LTSD(WINSIZE,window,5,lambda0=40)
    res_l,ltsds_l =  ltsd.compute_without_noise(sig_out_l)
    ltsd = LTSD(WINSIZE,window,5,lambda0=40)
    res_r,ltsds_r =  ltsd.compute_without_noise(sig_out_r)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ltsds_l)
    ax.plot(ltsds_r)
    plt.show()
    #print res_l
    #print res_r
    print "LTSD based vad is Done"

    sig_out_l = vad(res_l,sig_out_l,WINSIZE,window)
    sig_out_r = vad(res_l,sig_out_r,WINSIZE,window)
    print "vad is Done"
    """

    result = uniting_channles(sig_out_l, sig_out_r)
    write_signal(outfile, params, result)
    print "create wave file is Done"
    
    
