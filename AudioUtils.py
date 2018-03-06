import scipy
from scipy import signal
from scipy.io.wavfile import read
from scipy.fftpack import rfft
import matplotlib.pyplot as plt 
import numpy as np 


def getArrayFromAudio(audioFileName):
    print('****START of function getArrayFromAudio')
    
    #READ THE WAV FILE
    inputAudio = read(audioFileName)
    sampleRate = inputAudio[0]
    stereoAudio = inputAudio[1]
    
    #TRASFORM IN MONO
    monoAudio = np.int32(np.random.rand(len(stereoAudio),1))
    i = 0
    while i < len(stereoAudio):
        x = stereoAudio[i]
        monoAudio[i] = x[1]
        i += 1    
    
    #PRINT RESULT
    print('Sampling frequency: ',sampleRate)
    print('Stereo audio data: ',stereoAudio)
    print('Mono audio data: ',monoAudio)
    
    #PLOT THE AUDIO ARRAY
    plt.plot(monoAudio[:]) #plot the first 1024 samples
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title(audioFileName)
    plt.show() #display the plot

    print('****End of function getArrayFromAudio')
    return monoAudio, sampleRate

    
def getFrameArray(monoAudio, sampleRate, frameSize):
    print('****Start of method getFrameArray')
    
    resto = len(monoAudio)%frameSize #frame to discard to avoid error
    considLen = (len(monoAudio)-resto)/frameSize
    allFrame = np.int32(np.random.rand(np.int32(considLen),1)) #each position in a set of frames
    print('allFrame: ', allFrame)
    print('mono: ', monoAudio)
    
    i = 0
    y = 0
    while i < len(monoAudio):
        print(monoAudio[i:i+frameSize])
        x = monoAudio[i:i+frameSize]
        allFrame[y] = x 
        i += frameSize+1
        y += 1
        
    print('All Frame',allFrame)
    
    print('****End of method getFrameArray')          
    return allFrame
   
     
def getSpectrumFrameArray(arrayAudio):
    print('****Start of method getSpectrumFromArray')
    
    #COMPUTE FFT
    mags = abs(rfft(arrayAudio))
    mags = 20 * scipy.log10(mags)#Convert to dB
    mags -= max(mags)#Normalise to 0 dB max
    
    #PLOT GRAPH
    plt.plot(mags)
    plt.ylabel("Magnitude (dB)")
    plt.xlabel("Frequency Bin")
    plt.title("Spectrum")
    plt.show()
    
    print('****End of method getSpectrumFromArray')
    return mags

    
        