from face_detector import FaceDetector
from audio_registrator import AudioRegistrator
import queue
import os

if not os.path.isdir('registrazioni'):
    os.makedirs('registrazioni')

queue = queue.Queue(1)



fd = FaceDetector(queue)
ar = AudioRegistrator(queue)

fd.start()
ar.start()

