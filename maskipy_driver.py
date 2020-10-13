from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import argparse
from configparser import ConfigParser
import cv2
import numpy
import json


# press q to exit
def run_classifier(cfg):
    # start video capture
    video_capture = cv2.VideoCapture(0)
    casc_path = cfg['FACE_CASCADE']['FACE_CASCADE_XML']
    face_cascade = cv2.CascadeClassifier(casc_path)

    # load classifier model
    model = load_model(cfg['TRAINING']['MODEL_SAVE_PATH'])

    while True:
        ret, frame = video_capture.read()
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = face_cascade.detectMultiScale(frame_grayscale, scaleFactor=float(cfg['FACE_CASCADE']['SCALE_FACTOR']),
                                              minNeighbors=int(cfg['FACE_CASCADE']['MIN_NEIGHBOURS']), minSize=eval(cfg['FACE_CASCADE']['MIN_SIZE']),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
        # process face data
        for (x, y, w, h) in faces:
            x_inc, y_inc = int(w * 0.2), int(h * 0.2)
            try:
                frame_to_process = preprocess_input(
                    img_to_array(cv2.resize(frame[y - y_inc:y + h + y_inc, x - x_inc:x + w + x_inc], (224, 224))))
                results = model.predict(numpy.array([frame_to_process]), batch_size=int(cfg['TRAINING']['BATCH_SIZE']))
                predicted_label = json.loads(cfg['TRAINING']['MODEL_OUTPUT_LABELS'])[results.argmax()]
                label_data = (predicted_label, results.max(), eval(cfg['OUTPUT']['COLOR_MAP'])[predicted_label])

                label = "{}, Pr: {:.2f}".format(label_data[0], label_data[1])
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_data[2])
                cv2.rectangle(frame, (x, y), (x + w, y + h), label_data[2])

            except cv2.error:
                # ignore occasional resize errors
                pass

        cv2.imshow("maskipy", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CONFIG')
    parser.add_argument("-c", "--config", default='config.ini', help="config.ini file")
    args = vars(parser.parse_args())

    cfg = ConfigParser()
    cfg.read(args['config'])
    run_classifier(cfg)
