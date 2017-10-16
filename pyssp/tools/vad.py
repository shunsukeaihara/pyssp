#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
import wave
import tempfile
from pyssp.util import get_frame, add_signal, read_signal, separate_channels, uniting_channles
from pyssp.vad.ltsd import LTSD
import optparse

WINSIZE = 1024

def vad(vas, signal, winsize, window):
    out = np.zeros(len(signal), np.float32)
    for va in vas:
        for i in range(va[0], va[1]+2):
            add_signal(out, get_frame(signal, winsize, i)*window, winsize, i)
    return out

def write(param, signal):
    st = tempfile.TemporaryFile()
    wf = wave.open(st, 'wb')
    wf.setparams(params)
    s = np.int16(signal*32767.0).tostring()
    wf.writeframes(s)
    st.seek(0)
    print(st.read())

def read(fname, winsize):
    if fname == "-":
        wf = wave.open(sys.stdin, 'rb')
        n = wf.getnframes()
        audiobytes = wf.readframes(n)
        params = ((wf.getnchannels(), wf.getsampwidth(),
                   wf.getframerate(), wf.getnframes(),
                   wf.getcomptype(), wf.getcompname()))
        siglen = ((int)(len(audiobytes)/2/winsize) + 1) * winsize
        signal = np.zeros(siglen, np.float32)
        signal[0:len(str)/2] = np.float32(np.fromstring(audiobytes, np.int16))/32767.0
        return signal, params
    else:
        return read_signal(fname, winsize)


if __name__ == "__main__":
    """
    python vad.py -w WINSIZE -t THREATHOLD FILENAME
    """
    usage = "%python vad [-t THREASHOLD] [-w WINSIZE] [- s NOISETIME(ms)] "\
            "INPUTFILE \n if INPUTFILE is \"-\", read wave data from stdin"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("-w", type="int", dest="winsize", default=WINSIZE)
    parser.add_option("-t", type="int", dest="th", default=10)
    parser.add_option("-s", type="int", dest="ntime", default=300)

    (options, args) = parser.parse_args()
    windowsize = options.winsize

    fname = args[0]
    signal, params = read(fname, options.winsize)
    window = np.hanning(windowsize)
    ntime = options.ntime

    if params[0] == 1:
        ltsd = LTSD(windowsize, window, 5, lambda0=options.th)
        res, ltsds = ltsd.compute_with_noise(signal,
                                             signal[0:windowsize*int(params[2]/float(windowsize)/(1000.0/ntime))])
        write(params, vad(res, signal, windowsize, window))
    elif params[0] == 2:
        noise_sec = windowsize*int(params[2]/float(windowsize)/(1000.0/ntime))
        l, r = separate_channels(signal)
        ltsd_l = LTSD(windowsize, window, 5, lambda0=options.th)
        ltsd_r = LTSD(windowsize, window, 5, lambda0=options.th)
        l_out = vad(ltsd_l.compute_without_noise(l, noise_sec)[0], l, windowsize, window)
        r_out = vad(ltsd_r.compute_without_noise(r, noise_sec)[0], r, windowsize, window)
        out = uniting_channles(l_out, r_out)
        write(params, out)
