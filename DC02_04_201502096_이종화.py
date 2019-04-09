from __future__ import print_function

import sys
import wave

from io import StringIO

import alsaaudio
import colorama
import numpy as np
import librosa
from reedsolo import RSCodec, ReedSolomonError
from termcolor import cprint
from pyfiglet import figlet_format
import struct
import sounddevice as sd

my_num = "201502096"

HANDSHAKE_START_HZ = 4096
HANDSHAKE_END_HZ = 5120 + 1024

START_HZ = 1024
STEP_HZ = 256
BITS = 4

FEC_BYTES = 4

def stereo_to_mono(input_file, output_file):
    inp = wave.open(input_file, 'r')
    params = list(inp.getparams())
    params[0] = 1 # nchannels
    params[3] = 0 # nframes

    out = wave.open(output_file, 'w')
    out.setparams(tuple(params))

    frame_rate = inp.getframerate()
    frames = inp.readframes(inp.getnframes())
    data = np.fromstring(frames, dtype=np.int16)
    left = data[0::2]
    out.writeframes(left.tostring())

    inp.close()
    out.close()

def yield_chunks(input_file, interval):
    wav = wave.open(input_file)
    frame_rate = wav.getframerate()

    chunk_size = int(round(frame_rate * interval))
    total_size = wav.getnframes()

    while True:
        chunk = wav.readframes(chunk_size)
        if len(chunk) == 0:
            return

        yield frame_rate, np.fromstring(chunk, dtype=np.int16)

def dominant(frame_rate, chunk):
    window = np.hanning(len(chunk))
    
    w = np.fft.fft(np.multiply(chunk,window))
    
    freqs = np.fft.fftfreq(len(chunk))
    peak_coeff = np.argmax(np.abs(w))
    peak_freq = freqs[peak_coeff]
    return abs(peak_freq * frame_rate) # in Hz

def match(freq1, freq2):
    return abs(freq1 - freq2) < 20

def decode_bitchunks(chunk_bits, chunks):
    out_bytes = []

    next_read_chunk = 0
    next_read_bit = 0

    byte = 0
    bits_left = 8
    while next_read_chunk < len(chunks):
        can_fill = chunk_bits - next_read_bit
        to_fill = min(bits_left, can_fill)
        offset = chunk_bits - next_read_bit - to_fill
        byte <<= to_fill
        shifted = chunks[next_read_chunk] & (((1 << to_fill) - 1) << offset)
        byte |= shifted >> offset;
        bits_left -= to_fill
        next_read_bit += to_fill
        if bits_left <= 0:

            out_bytes.append(byte)
            byte = 0
            bits_left = 8

        if next_read_bit >= chunk_bits:
            next_read_chunk += 1
            next_read_bit -= chunk_bits

    return out_bytes

def decode_file(input_file, speed):
    wav = wave.open(input_file)
    if wav.getnchannels() == 2:
        mono = StringIO()
        stereo_to_mono(input_file, mono)

        mono.seek(0)
        input_file = mono
    wav.close()

    offset = 0
    for frame_rate, chunk in yield_chunks(input_file, speed / 2):
        dom = dominant(frame_rate, chunk)
        print("{} => {}".format(offset, dom))
        offset += 1

def extract_packet(freqs):
    #print("           ",len(freqs))
    freqs = freqs[::2]
    bit_chunks = [int(round((f - START_HZ) / STEP_HZ)) for f in freqs]
    bit_chunks = [c for c in bit_chunks[1:] if 0 <= c < (2 ** BITS)]
    #for i in range(len(bit_chunks)):
    #    print(bit_chunks[i])
    
    return bytearray(decode_bitchunks(BITS, bit_chunks))

def display(s):
    cprint(figlet_format(s.replace(' ', '   '), font='doom'), 'yellow')

def hzToSound(freq, tlen, Fs = 44100):

    #Fs = 44100
    Ts = 1/Fs
    t = np.arange(0,tlen,Ts)
    signal= np.sin(2*np.pi*freq*t)
    return signal

def strToByteStream(line):
    return RSCodec(FEC_BYTES).encode(bytearray(bytes(line, 'utf-8')))

def ByteStreamToSignal(ByteStream):
    signal = np.zeros(0)
    signal = np.append(signal,hzToSound(HANDSHAKE_START_HZ,0.1))
    
    #byte_list = []
    for i in range(len(ByteStream)):
        char = ByteStream[i]
        #print(char)
        s1 = char//(2**4)
        s2 = char%(2**4)
        #print(s1,s2,s3,s4)
        signal = np.append(signal,hzToSound(START_HZ + STEP_HZ*(s1),0.1))
        signal = np.append(signal,hzToSound(START_HZ + STEP_HZ*(s2),0.1))
        #signal = np.append(signal,hzToSound(START_HZ + STEP_HZ*(s3),0.1))
        #signal = np.append(signal,hzToSound(START_HZ + STEP_HZ*(s4),0.1))
    signal = np.append(signal,hzToSound(HANDSHAKE_END_HZ,0.1))
    
    return signal
'''
def byteListToSignal(byte_list):
    
    signal = np.zeros(0)
    signal = np.append(signal,hzToSound(HANDSHAKE_START_HZ,0.1))
    
    for i in range(len(byte_list)):
        signal = np.append(signal,hzToSound(START_HZ + STEP_HZ*(byte_list[i]),0.1))
    signal = np.append(signal,hzToSound(HANDSHAKE_END_HZ,0.1))
    return signal
'''
'''
def test(byte_stream):
    frame_rate = 44100
    if my_num in byte_stream:
        factors = byte_stream.split(my_num)
        byte_stream = factors[0] + factors[1]
        print("no num:",byte_stream)
        #byte_stream2 = bytearray(bytes(byte_stream, 'utf-8'))
        byte_stream = strToByteStream(byte_stream)
        print("bytes:",byte_stream)
        signal = ByteStreamToSignal(byte_stream)
        sd.play(signal,frame_rate)
        from time import sleep
        sleep(len(signal)/frame_rate)
'''     
def RunSoundWithoutID(byte_stream, frame_rate=44100):
    factors = byte_stream.split(my_num)
    byte_stream = ("").join(factors)
    #print(byte_stream)
    #byte_stream = factors[0] + factors[1]
    byte_stream = strToByteStream(byte_stream)
    signal = ByteStreamToSignal(byte_stream)
   
    sd.play(signal,frame_rate)
    from time import sleep
    sleep(len(signal)/frame_rate)
    #librosa.output.write_wav("./sound.wav",signal, frame_rate)

def listen_linux(frame_rate=44100, interval=0.1):

    mic = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device="default")
    mic.setchannels(1)
    mic.setrate(44100)
    mic.setformat(alsaaudio.PCM_FORMAT_S16_LE)

    num_frames = int(round((interval / 2) * frame_rate))
    mic.setperiodsize(num_frames)
    print("start...")

    in_packet = False
    packet = []

    while True:
        l, data = mic.read()
        
        if not l:
            continue
        chunk = np.fromstring(data, dtype=np.int16)
        dom = dominant(frame_rate, chunk)
        #if in_packet:
        #    print(dom)
        if in_packet and match(dom, HANDSHAKE_END_HZ):

            byte_stream = extract_packet(packet)
            try:
                #print(byte_stream)
                byte_stream = RSCodec(FEC_BYTES).decode(byte_stream)
                #print(byte_stream)
                byte_stream = byte_stream.decode("utf-8")
                display(byte_stream)

                if my_num in byte_stream:
                    RunSoundWithoutID(byte_stream)
            except ReedSolomonError as e:
                pass
                #print("{}: {}".format(e, byte_stream))

            packet = []
            in_packet = False
        elif in_packet:
            packet.append(dom)
        elif match(dom, HANDSHAKE_START_HZ):
            in_packet = True

if __name__ == '__main__':
    colorama.init(strip=not sys.stdout.isatty())
    #Byte_stream = strToByteStream('hi')
    #print(Byte_stream)
    #print(len(Byte_stream))
    
    #print(RSCodec(FEC_BYTES).encode(bytearray(bytes('hi', 'utf-8'))))
    #print(struct.unpack('!6b',Byte_stream))
    
    #print(Byte_stream.decode("utf-8"))
    #decode_file(sys.argv[1], float(sys.argv[2]))
    #RunSoundWithoutID("my201502096id")
    listen_linux()
    #test("hi2096hi")
