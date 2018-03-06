import scipy
from scipy import signal
from scipy.io.wavfile import read
from scipy.fftpack import rfft
import matplotlib.pyplot as plt 
import numpy as np 

'''
Input: path for the file
Output: array of the mono information and the sample rate 
'''
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

'''
Input: array of the mono info, frame size choosen
Output: array of object containing frame info [[[a b c]][[d e f]]...] example for frame of size 3
'''    
def getFrameArray(monoAudio, sampleRate, frameSize):
    print('****Start of method getFrameArray')
    
    #resto = len(monoAudio)%frameSize #frame to discard to avoid error
    #considLen = (len(monoAudio)-resto)/frameSize
    #allFrame = np.int32(np.random.rand(np.int32(considLen),1)) #each position in a set of frames
    #print('allFrame: ', allFrame)
    allFrame=[]
    
    print('Mono: ', monoAudio)
    print('Sample rate: ', sampleRate)
    
    i = 0
    while i < len(monoAudio):
        x = monoAudio[i:i+frameSize]
        allFrame.append(x)
        i += frameSize+1
        
    print('All Frame[0]',allFrame[0])
    
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

    
        