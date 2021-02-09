# Multimodal Audio-Textual Emotion Recognition Neural Network

Since emotions are expressed through a combination of verbal and non-verbal channels, a joint analysis of speech and gestures is required to understand
expressive human communication.

## Data
The IEMOCAP (Interactive emotional dyadic motion capture) database dataset can be downloaded from [IEMOCAP_link](https://sail.usc.edu/iemocap/).

This database was recorded from ten actors in dyadic sessions with markers on the face, head, and hands, which provide detailed information about their facial expression and
hand movements during scripted and spontaneous spoken communication scenarios. The actors performed selected emotional scripts and also improvised hypothetical
scenarios designed to elicit specific types of emotions (happiness, anger, sadness, frustration and neutral state). 

The corpus contains approximately twelve hours
of data. The detailed motion capture information, the interactive setting to elicit authentic emotions, and the size of the database make this corpus a valuable addition
to the existing databases in the community for the study and modeling of multimodal and expressive human communication.
 
## Overview
Direct link to the master thesis describing the approach code [Thesis_link](https://www.politesi.polimi.it/bitstream/10589/143008/3/PATHOSnet.pdf).

An approach for emotion recognition leveraging Neural Networks, which combines audio and text analysis. The purpose of this
work is to build a system capable of recognising different emotions combining acoustic and textual information, and show that this approach outperforms
systems based on the separate analysis of these two modalities. 

The proposed model, is built and evaluated on the IEMOCAP corpus, which offers realistic audio recording and transcription of sentences with emotional content. The model reaches a test accuracy of 73:5%, while the preceding best score of an automatic system, leveraging the same modalities, was 69:7% and human listeners reach 70% (on the four considered emotions).

![Alt text](/structure.JPG?raw=true =250x)





