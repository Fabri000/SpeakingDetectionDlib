import pyaudio
import wave
import time
from threading import Thread
from queue import Queue
import _queue

class AudioRegistrator(Thread):
    
    def __init__(self,queue,chunk=1024,format=pyaudio.paInt16,channels=2,frequency =44100):
        super().__init__()

        self.chunk = chunk
        self.format = format
        self.channels = channels
        self.frequency = frequency

        self.queue = queue
        self.audio = pyaudio.PyAudio()

        

    def run(self):
        
        start_recording = False
        proceed = True
        
        while proceed:
            
            frames = []  # List of audio portion
            
            # signal for the mouth open
            signal = self.queue.get()
            if(signal == 'Face detected'):
                stream = self.audio.open(format=self.format,
                    channels=self.channels,
                    rate=self.frequency,
                    input=True,
                    frames_per_buffer=self.chunk)
                
                while True:
                    # Register audio frame
                    data = stream.read(self.chunk)

                    # Add audio frame to the list
                    frames.append(data)

                    # Stop recording signal with the mouth closed detection
                    try:
                        signal = self.queue.get(False)

                        if (signal == 'Open') and  not(start_recording):
                            start_recording = True
                            frames = frames[len(frames)-7:]

                        elif signal == 'Closed' and start_recording:
                            
                            start_recording = False

                            stream.stop_stream()
                            stream.close()

                            # Create wav file with output
                            cur_time = time.time()
                            file_audio = wave.open(f"registrazioni/registrazione-{cur_time}.wav", "wb")

                            # Wav file settings
                            file_audio.setnchannels(self.channels)
                            file_audio.setsampwidth(self.audio.get_sample_size(self.format))
                            file_audio.setframerate(self.frequency)

                            # Write on file the output
                            file_audio.writeframes(b''.join(frames))
                            
                            file_audio.close()

                            break

                        elif signal == 'End':
                            proceed = False
                            break
                        
                    except _queue.Empty:
                        print()
                    
                