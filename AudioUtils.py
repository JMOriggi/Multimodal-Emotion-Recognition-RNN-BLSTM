#Classe contente tutte le funzionalità legate al funzionamento della NN.
#Propone vari modelli di NN, e permette l'estrazione per il salvataggio dei weight che definiscono la rete.


from scipy import signal
from scipy.io import wavfile




class AudioUtils:
    
    def __init__(self):
        print('****Initiate Class AudioUtils')
    
    
    def getSpectrumFromAudio(self,In):
        print('****Start of method getSpectrumFromAudio')
        #filename = "file.wav"
        #audData = scipy.io.wavfile.read(filename)
        #print(audData)
        
        #sample_rate, samples = wavfile.read('file.wav')
        #frequencies, times, spectogram = signal.spectrogram(samples, sample_rate)
        
        #plt.imshow(spectogram)
        #plt.pcolormesh(times, frequencies, spectogram)
        #plt.ylabel('Frequency [Hz]')
        #plt.xlabel('Time [sec]')
        #plt.show()
        print('****End of method getSpectrumFromAudio')
    
        
    def plotSpectrum(self,In):
        print('****Start of method plotSpectrum')
        print('****End of method plotSpectrum')
    
        
    def getNextFrame(self,Model):
        print('****Start of method getNextFrame')
        print('****End of method getNextFrame')  

