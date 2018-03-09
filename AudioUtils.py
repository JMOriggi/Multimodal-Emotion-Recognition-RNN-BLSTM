import scipy
from scipy import signal
from scipy.io.wavfile import read
from scipy.fftpack import rfft
from scipy.fftpack import fft
import matplotlib.pyplot as plt 
import numpy as np 

#Input: path for the file
#Output: array of the mono information and the sample rate 
def getArrayFromAudio(audioFileName):
    print('****START of function getArrayFromAudio')
    
    #READ THE WAV FILE
    inputAudio = read(audioFileName)
    sampleRate = inputAudio[0]
    stereoAudio = inputAudio[1]
    
    #TRASFORM IN MONO
    monoAudio = (stereoAudio[:,0] + stereoAudio[:,1]) / 2  
    
    #PRINT RESULT
    print('Sampling frequency: ',sampleRate)
    print('Stereo audio data (first 1024 samples): ',stereoAudio[0:1024])
    print('Mono audio data (first 1024 samples): ',monoAudio[0:1024])
    print('Mono audio matrix structure: ',monoAudio.shape)
    
    #PLOT THE AUDIO ARRAY
    plt.plot(monoAudio)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title(audioFileName)
    plt.show() #display the plot

    print('****End of function getArrayFromAudio')
    return monoAudio, sampleRate


#INPUT: array of the mono info, sampleRate, frame size choosen
#Output: list of chunks containing frame info, [[[a b c]][[d e f]]...] example for frame of size 3 
def getFrameArray(monoAudio, sampleRate, frameSize):
    print('****Start of method getFrameArray')
    
    #DIVIDE ARRAY IN CHUNKS: each chunks of a frame size of samples
    allFrame=[]
    i = 0
    while i < len(monoAudio):
        x = monoAudio[i:i+frameSize]
        allFrame.append(x)
        i += frameSize+1
        
    #PRINT RESULTS
    print('Mono first 1024 samples: ', monoAudio[0:1024])
    print('Sample rate: ', sampleRate)
    print('Frame size: ', frameSize)     
    print('All Frame[0]',allFrame[0])
    print('size row: ',len(allFrame))
    print('size collums: ',len(allFrame[0]))
    
    print('****End of method getFrameArray')          
    return allFrame
   
   
#INPUT: the frame chunks list array     
#OUTPUT: the fft list of each frame chunks
def getSpectrumFrameArray(allFrame):
    print('****Start of method getSpectrumFromArray')
    
    #COMPUTE FFT: for each frame window chunks we will obtain a magnitude fft chunks
    allFrameFFT = []
    mags = []
    i = 0
    while i < len(allFrame):
        mags = abs(rfft(allFrame[i]))
        mags = 20 * scipy.log10(mags)#Convert to dB
        mags -= max(mags)#Normalise to 0 dB max
        allFrameFFT.append(mags)
        i += 1
        
    
    #PLOT GRAPH: example for first frame window
    plt.plot(allFrameFFT[0])
    plt.ylabel("Magnitude (dB)")
    plt.xlabel("Frequency Bin")
    plt.title("Spectrum")
    plt.show()
    
    #allFrameFFT = [arr.tolist() for arr in allFrameFFT]
    #print('####: ',allFrameFFT)
    print('size row: ',len(allFrameFFT))
    print('size collums: ',len(allFrameFFT[0]))
    
    
    print('****End of method getSpectrumFromArray')
    return allFrameFFT

    
        