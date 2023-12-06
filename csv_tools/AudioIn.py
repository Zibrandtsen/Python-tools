import numpy as np
import pyaudio
import time
import wave
from scipy.signal import butter, lfilter, find_peaks
from scipy.fftpack import rfft, rfftfreq

from multiprocessing import Process, Queue, Value

from csv_tools import writeCSV, clearFile
from DTMF_module import findDTMF

class AudioIn:
    #TODO: Add so analasis can run without duration limit
    def __init__(self, chunk_size=512, window_size = 16, min_analasis_peak = 50, read_duration=5, save_wav=False):
        self.CHUNK_SIZE = chunk_size
        self.min_analasis_peak = min_analasis_peak
        self.SAMPLE_RATE = 44100
        self.read_duration = read_duration
        self.save_wav = save_wav
        self.save_dir = "wave_files/"

        self.FORMAT = np.int16
        self.P_FORMAT = pyaudio.paInt16

        # self.window_list = np.zeros((window_size, chunk_size))
        self.window_list = []
        for i in range(window_size-1):
            self.window_list.append(np.zeros(chunk_size))
        
        self.audio_in = Queue()
        self.recoording = Value('b', True) 

        # Used to find hex values from stream
        self.input_buffer = []
        self.temp_storage = []
        self.rx_values = []

    def readStream(self):
        """
        Reads audio stream from the input device and puts the audio data into a queue {self.audio_in} for processing.

        Returns:
            None
        """
        if self.save_wav:
            data_list = []

        output_path = "output.csv"
        clearFile(output_path)

        p = pyaudio.PyAudio()
        stream = p.open(format=self.P_FORMAT,
                        channels=1,
                        rate=self.SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK_SIZE)

        while self.recoording.value:
            data = stream.read(self.CHUNK_SIZE)
            result = np.frombuffer(data, dtype=self.FORMAT)
            self.audio_in.put(result)

            if self.save_wav:
                data_list.append(data)
        
        if self.save_wav:
            self.save2Wav(b''.join(data_list))

        stream.stop_stream()
        stream.close()
        p.terminate()

    def readWav(self, file_name: str):
        """
        Reads a WAV file and puts the audio data into a queue {self.audio_in} for processing.

        Args:
            file_name (str): The name of the WAV file to be read.

        Returns:
            None
        """
        wf = wave.open(self.save_dir + file_name, 'rb')
        data = wf.readframes(self.CHUNK_SIZE)
        while data != b'':
            result = np.frombuffer(data, dtype=self.FORMAT)
            self.audio_in.put(result)
            data = wf.readframes(self.CHUNK_SIZE)
        wf.close()
        self.recoording.value = False

    def run(self, live=True):
        """
        Runs the audio processing and analysis.

        Args:
            live (bool, optional): Determines whether to process live audio or from a WAV file. 
                Defaults to True.

        Returns:
            None
        """
        output_path = "output.csv"
        result_path = "results.csv"
        clearFile(output_path)
        clearFile(result_path)
        if live:
            start_P = Process(target=self.readStream,)
        else:
            start_P = Process(target=self.readWav, args=("output.wav",))
        start_P.start()
        
        print("Starting...")
        start_time = time.time()
        while self.recoording.value or not self.audio_in.qsize() == 0:
            
            # Handle input and window
            queue_in = self.audio_in.get()
            self.window_list.append(queue_in)
            window_array = np.array(self.window_list).flatten()
            self.window_list.pop(0)

            # Analyze window
            peaks_f, peaks_a = self.analyze(window_array)
            hex_val, sort_freqs = findDTMF(peaks_f)

            # Find message
            self.findHexFromQueue(hex_val)
            # print(self.rx_values)

            writeCSV(output_path, [hex_val, sort_freqs, peaks_f, peaks_a])
            writeCSV(result_path, self.rx_values)

            if time.time() - start_time > self.read_duration:
                self.recoording.value = False
                break

        print("Done!")
        start_P.join()
        start_P.terminate()

    def analyze(self, raw_signal : np.array, PASS_BAND = [650, 1850]):
        """
        Analyzes the raw signal using a bandpass filter and returns the peak frequencies and amplitudes.

        Parameters:
        - raw_signal (np.array): The raw signal to be analyzed.
            - Should be a 1D array of the sampels.
        - PASS_BAND (list): The pass band for the bandpass filter. Default is [650, 1850].

        Returns:
        - list: A list containing the peak frequencies and amplitudes.
        """
        filtered_sample_list = self.butterBandpassFilter(raw_signal, PASS_BAND)

        ## FFT begins
        N_SAMPELS = len(filtered_sample_list)
        sample_frequencies = rfftfreq(N_SAMPELS, d=1.0/self.SAMPLE_RATE)                 # Discrete Fourier Transform sample frequencies
        norm_amplitude = 2*np.abs(rfft(filtered_sample_list))/N_SAMPELS     # den reale amplitude, tilpasset til frekvenserne

        # Find peak frequencies
        peaks_placement, _ = find_peaks(norm_amplitude, height=self.min_analasis_peak)

        freq_peak, amp_peak = [], []
        for i in peaks_placement:
            freq_peak.append(sample_frequencies[i])
            amp_peak.append(norm_amplitude[i])

        return [freq_peak, amp_peak]

    def findHexFromQueue(self, value : int, THRESHOLD=0.6, BUFFER_SIZE=4, MIN_HEXS_IN_BUFF=6):
        """
        Finds hexadecimal values from the input queue based on THRESHOLD, BUFFER_SIZE and MIN_HEXS_IN_ROW

        Args:
            value (int): The value to be added to the input buffer.
            THRESHOLD (float, optional): Used to check if 60% (default) of a hex value is pressent in the input buffer. 
            BUFFER_SIZE (int, optional): The size of the input buffer. Defaults to 10.
            MIN_HEXS_IN_BUFF (int, optional): The minimum number of a hex value before added to result.

        Returns:
            None
        """

        self.input_buffer.append(value)
        checkList = []

        # Generates a list of -1's
        if self.input_buffer.count(-1) == BUFFER_SIZE:
            for i in range(len(self.input_buffer)):
                checkList.append(-1)

        # If 3 consecutive -1's are found
        if self.input_buffer == checkList:
            self.temp_set = list(set(self.temp_storage))

            if len(self.temp_set) > 0:
                for category in self.temp_set:
                    count = self.temp_storage.count(category)
                    if category != None and count >= len(self.temp_storage) * THRESHOLD and count >= MIN_HEXS_IN_BUFF:
                        self.rx_values.append(category)
                        self.temp_set = []
                self.temp_storage = []

        elif value != -1:
            self.temp_storage.append(value)

        if len(self.input_buffer) >= BUFFER_SIZE:
            self.input_buffer.pop(0)

    def butterBandpassFilter(self, data, pass_band : list, order=5):
        """
        Apply a Butterworth bandpass filter to the input data.

        Parameters:
        - data: The input data to be filtered.
        - pass_band: A list containing the lower and upper cutoff frequencies of the pass band.
        - order: The order of the Butterworth filter (default is 5).

        Returns:
        - y: The filtered output data.

        Reference:
        - https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
        """
        b, a = butter(order, pass_band, fs=self.SAMPLE_RATE, btype='bandpass')
        y = lfilter(b, a, data)
        return y
    
    def save2Wav(self, data : bytes, filename='output.wav'):
        """
        Save the audio data as a WAV file.

        Args:
            data (bytes): The audio data to be saved.
            filename (str, optional): The name of the output WAV file. Defaults to 'output.wav'.
        """
        with wave.open(self.save_dir + filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.FORMAT().itemsize)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(data)

if __name__ == "__main__":
    audio = AudioIn(min_analasis_peak=50, save_wav=True)
    audio.run(True)
    # audio.run(False)