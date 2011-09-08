#!/usr/bin/env python
# -*- coding: utf-8 -*-
#reference http://kaiseki-web.lhd.nifs.ac.jp/wiki/index.php/Python_%E3%81%AB%E3%82%88%E3%82%8B%E3%82%AA%E3%83%BC%E3%83%87%E3%82%A3%E3%82%AA%E5%87%A6%E7%90%86

import wave
import numpy as np
import scipy as sp
from itertools import izip

def read_signal(filename, winsize):
    wf=wave.open(filename,'rb')
    n=wf.getnframes()
    str=wf.readframes(n)
    params = ((wf.getnchannels(), wf.getsampwidth(),
               wf.getframerate(), wf.getnframes(),
               wf.getcomptype(), wf.getcompname()))
    siglen=((int )(len(str)/2/winsize) + 1) * winsize
    signal=sp.zeros(siglen, sp.float32)
    signal[0:len(str)/2] = sp.float32(sp.fromstring(str,sp.int16))/32767.0
    return [signal, params]


def get_frame(signal, winsize, no):
    shift=winsize/2
    start=no*shift
    end = start+winsize
    return signal[start:end]


def add_signal(signal, frame, winsize, no ):
    shift=winsize/2
    start=no*shift
    end=start+winsize
    signal[start:end] = signal[start:end] + frame


def write_signal(filename, params ,signal):
    wf=wave.open(filename,'wb')
    wf.setparams(params)
    s=sp.int16(signal*32767.0).tostring()
    wf.writeframes(s)


def get_window(winsize,no):
    shift=winsize/2
    s = no*shift
    return (s, s+winsize)


def separate_channels(signal):
    return signal[0::2],signal[1::2]


def uniting_channles(leftsignal,rightsignal):
    ret=[]
    for i,j in izip(leftsignal,rightsignal):
        ret.append(i)
        ret.append(j)
    return np.array(ret,sp.float32)

def compute_avgamplitude(signal,winsize,window):
    windownum = len(signal)/(winsize/2) - 1
    avgamp = sp.zeros(winsize)
    for l in xrange(windownum):
        avgamp += sp.absolute(sp.fft(get_frame(signal, winsize,l) * window))
    return avgamp/float(windownum)

def compute_avgpowerspectrum(signal,winsize,window):
    windownum = len(signal)/(winsize/2) - 1
    avgpow = sp.zeros(winsize)
    for l in xrange(windownum):
        avgpow += sp.absolute(sp.fft(get_frame(signal, winsize,l) * window))**2.0
    return avgpow/float(windownum)

def sigmoid(x, x0, k, a):
    y = k * 1 / (1 + np.exp(-a*(x-x0)))
    return y
