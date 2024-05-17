from face_detector import FaceDetector
from audio_registrator import AudioRegistrator
import queue

queue = queue.Queue(1)

fd = FaceDetector(queue)
ar = AudioRegistrator(queue)

fd.start()
ar.start()

fd.join()
ar.join()