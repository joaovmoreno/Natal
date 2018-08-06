import urllib
import scipy.io.wavfile
import pydub
import matplotlib.pyplot as plt
from numpy import fft as fft
import numpy as np

# a temp folder for downloads
temp_folder = "/Users/home/Desktop/"

# spotify mp3 sample file
web_file = "http://p.scdn.co/mp3-preview/35b4ce45af06203992a86fa729d17b1c1f93cac5"

# download file
urllib.urlretrieve(web_file, temp_folder + "file.mp3")
# read mp3 file
mp3 = pydub.AudioSegment.from_mp3(temp_folder + "file.mp3")
# convert to wav
mp3.export(temp_folder + "file.wav", format="wav")
# read wav file
rate, audData = scipy.io.wavfile.read(temp_folder + "file.wav")

# the sample rate is the number of bits of infomration recorded per second
print(rate)
print(audData)

# wav bit type the amount of information recorded in each bit often 8, 16 or 32 bit
audData.dtype

# wav length
audData.shape[0] / rate

# wav number of channels mono/stereo
audData.shape[1]
# if stereo grab both channels
channel1 = audData[:, 0]  # left
channel2 = audData[:, 1]  # right



# Energy of music
np.sum(channel1.astype(float) ** 2)
# this can be infinite and depends on the length of the music of the loudness often talk about power
# power - energy per unit of time
1.0 / (2 * (channel1.size) + 1) * np.sum(channel1.astype(float) ** 2) / rate

# save wav file
scipy.io.wavfile.write(temp_folder + "file2.wav", rate, audData)
# save a file at half and double speed
scipy.io.wavfile.write(temp_folder + "file2.wav", rate / 2, audData)
scipy.io.wavfile.write(temp_folder + "file2.wav", rate * 2, audData)

# save a single channel
scipy.io.wavfile.write(temp_folder + "file2.wav", rate, channel1)

# averaging the channels damages the music
mono = np.sum(audData.astype(float), axis=1) / 2
scipy.io.wavfile.write(temp_folder + "file2.wav", rate, mono)



time = np.arange(0, float(audData.shape[0]), 1) / rate

# plot amplitude (or loudness) over time
plt.figure(1)
plt.subplot(211)
plt.plot(time, channel1, linewidth=0.02, alpha=0.7, color='#ff7f00')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(212)
plt.plot(time, channel2, linewidth=0.02, alpha=0.7, color='#ff7f00')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig(temp_folder + 'ampiltude.png', bbox_inches='tight')

# Frequency (pitch) over time

# a fourier transform breaks the sound wave into series of waves that make up the main sound wave
# each of these waves will have its own amplitude (volume) and frequency. The frequency is the length over which the wave repeats itself. this is known as the pitch of the sound



fourier = fft.fft(channel1)

plt.figure(1, figsize=(8, 6))
plt.plot(fourier, color='#ff7f00')
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.savefig(temp_folder + 'fft.png', bbox_inches='tight')

# the fourier is symetrical due to the real and imaginary soultion. only interested in first real solution
n = len(channel1)
fourier = fourier[0:(n / 2)]

# scale by the number of points so that the magnitude does not depend on the length
fourier = fourier / float(n)

# calculate the frequency at each point in Hz
freqArray = np.arange(0, (n / 2), 1.0) * (rate * 1.0 / n);

plt.figure(1, figsize=(8, 6))
plt.plot(freqArray / 1000, 10 * np.log10(fourier), color='#ff7f00', linewidth=0.02)
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power (dB)')
plt.savefig(temp_folder + 'frequencies.png', bbox_inches='tight')

# plot spectogram
# the function calculates many fft's over NFFT sized blocks of data
# increasing NFFT gives you a more detail across the spectrum range but decreases the samples per second
# the sampling rate used determines the frequency range seen always 0 to rate/2

plt.figure(2, figsize=(8, 6))
plt.subplot(211)
Pxx, freqs, bins, im = plt.specgram(channel1, Fs=rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar = plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity dB')
plt.subplot(212)
Pxx, freqs, bins, im = plt.specgram(channel2, Fs=rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar = plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity (dB)')
# plt.show()
plt.savefig(temp_folder + 'spectogram.png', bbox_inches='tight')

# Larger Window Size value increases frequency resolution
# Smaller Window Size value increases time resolution
# Specify a Frequency Range to be calculated for using the Goertzel function
# Specify which axis to put frequency


Pxx, freqs, timebins, im = plt.specgram(channel2, Fs=rate, NFFT=1024, noverlap=0, cmap=plt.get_cmap('autumn_r'))

channel1.shape
Pxx.shape
freqs.shape
timebins.shape
np.min(freqs)
np.max(freqs)
np.min(timebins)
np.max(timebins)

np.where(freqs == 10034.47265625)
MHZ10 = Pxx[233, :]
plt.figure(figsize=(8, 6))
plt.plot(timebins, MHZ10, color='#ff7f00')
plt.savefig(temp_folder + 'MHZ10.png', bbox_inches='tight')
