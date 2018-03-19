import scipy
from scipy import signal
from scipy.io.wavfile import read
from scipy.fftpack import rfft
from scipy.fftpack import fft
import matplotlib.pyplot as plt 
import numpy as np 

#INPUT: path for the file
#OUTPUT: array of the mono information and the sample rate 
def getArrayFromAudio(audioFileName):
    print('****START of function getArrayFromAudio')
    
    #READ THE WAV FILE
    inputAudio = read(audioFileName)
    sampleRate = inputAudio[0]
    stereoAudio = inputAudio[1]
    
    #TRASFORM IN MONO: no need because audio already in mono
    monoAudio = stereoAudio
    #monoAudio = (stereoAudio[:,0] + stereoAudio[:,1]) / 2
    
    #PRINT RESULT
    '''print('Sampling frequency: ',sampleRate)
    #print('Stereo audio data (first 100 samples): ',stereoAudio[0:100])
    print('Mono audio data (first 100 samples): ',monoAudio[0:1000])
    print('Mono audio matrix structure: ',monoAudio.shape)'''
    
    #PLOT THE AUDIO ARRAY
    '''plt.plot(monoAudio)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title(audioFileName)
    plt.show() #display the plot'''

    print('****End of function getArrayFromAudio\n')
    return monoAudio, sampleRate

#GET THE FREQUENCY ARRAY: [timestep [freqs amplitude]]
def getFreqArray(monoAudio, sampleRate): 
    print('****Start of method getSpectrumFromArray')
    
    #COMPUTE SPECTROGRAM
    fft, freqsBins, timeBins, im = plt.specgram(monoAudio, Fs=sampleRate, NFFT=320, cmap=plt.get_cmap('autumn_r'))
    print('shape fft: ', fft.shape)
    print('shape timeBins ', timeBins.shape)
    print('shape freqsBins: ', freqsBins.shape)
    
    #PRINT SPECTROGRAM
    cbar=plt.colorbar(im)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    cbar.set_label('Intensity (dB)')
    plt.show()
    
    #RESHAPE FREQ ARRAY: to prepare it for the NN, row=timestep collumns=freq values
    i = 0
    X = np.full((len(fft[0]), len(fft)), 0)
    while i < len(fft):
        y = 0
        while y < len(fft[0]):
            X[y][i] = fft[i][y]
            y+=1
        i+=1
    print('New shape Pxx: ', X.shape) 
    
    print('****End of method getSpectrumFromArray\n')
    return np.asarray(X)
 

#INPUT: array of the mono info, sampleRate, frame size choosen
#Output: list of chunks containing frame info, [[[a b c]][[d e f]]...] example for frame of size 3 
def getFrameArray(monoAudio, sampleRate):
    print('****Start of method getFrameArray')
    
    #DIVIDE ARRAY IN CHUNKS: each chunks of a frame size of samples
    frameSize = 320 #Frame size setted to 320samples hat correspond to chunks of 20ms
    allFrame=[]
    i = 0
    while i < len(monoAudio):
        x = monoAudio[i:i+frameSize]
        allFrame.append(x)
        i += frameSize+1
        
    #PRINT RESULTS
    '''print('Mono first 100 samples: ', monoAudio[0:100])
    print('Sample rate: ', sampleRate)
    print('Frame size: ', frameSize)     
    print('All Frame[0]',allFrame[0])
    print('size row: ',len(allFrame))
    print('size collums: ',len(allFrame[0]))'''
    
    print('****End of method getFrameArray\n')          
    return allFrame
   
   
#INPUT: the frame chunks list array     
#OUTPUT: the fft list of each frame chunks, same format of the input
def getSpectrumFrameArray(allFrame):
    print('****Start of method getSpectrumFromArray')
    
    #COMPUTE FFT: for each frame window chunks we will obtain a magnitude fft chunks
    allFrameFFT = []
    mags = []
    i = 0
    while i < len(allFrame)-1: #-1 beacause the last chunks may be shorter than the others
        mags = abs(rfft(allFrame[i]))
        #mags = 20 * scipy.log10(mags)#Convert to dB
        #mags -= max(mags)#Normalise to 0 dB max
        allFrameFFT.append(mags)
        i += 1
        
    
    #PLOT GRAPH: example for first frame window
    plt.plot(allFrameFFT[0])
    plt.ylabel("Magnitude (dB)")
    plt.xlabel("Frequency Bin")
    plt.title("Spectrum")
    plt.show()
    
    print('allFrameFFT: ',allFrameFFT)
    print('allFrameFFT size row: ',len(allFrameFFT))
    print('allFrameFFT size collums: ',len(allFrameFFT[0]))
    
    
    print('****End of method getSpectrumFromArray\n')
    return np.asarray(allFrameFFT)


       