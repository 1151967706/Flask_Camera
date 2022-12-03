import os
import cv2
from base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            start = cv2.getTickCount()
            _, img = camera.read()

            # the fps put in image
            stop = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (stop - start)
            fps = '{}: {:.3f}'.format('FPS', fps)
            (fps_w, fps_h), baseline = cv2.getTextSize(fps, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (2, 20 - fps_h - baseline), (2 + fps_w, 18), color=(0, 0, 0), thickness=-1)
            cv2.putText(img, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255, 255, 255), thickness=2)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()