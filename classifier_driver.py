from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import config as cfg
import cv2
import numpy

# start video capture
video_capture = cv2.VideoCapture(0)
cascPath = cfg.face_casc_path
faceCascade = cv2.CascadeClassifier(cascPath)

# load classifier model
model = load_model(cfg.mask_model)


# press q to exit
def run_classifier():
    while True:
        ret, frame = video_capture.read()
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = faceCascade.detectMultiScale(frame_grayscale, scaleFactor=cfg.scaleFactor,
                                             minNeighbors=cfg.minNeighbors, minSize=(30, 30),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        # process face data
        for (x, y, w, h) in faces:
            x_inc, y_inc = int(w * 0.2), int(h * 0.2)
            try:
                frame_to_process = preprocess_input(
                    img_to_array(cv2.resize(frame[y - y_inc:y + h + y_inc, x - x_inc:x + w + x_inc], (224, 224))))
                results = model.predict(numpy.array([frame_to_process]), batch_size=cfg.BATCH_SIZE)
                improperly_masked, not_masked, properly_masked = results[0]
                label_data = [('Improperly Masked', improperly_masked, (255, 255, 0)),
                              ('Unmasked', not_masked, (0, 0, 255)),
                              ('Properly Masked', properly_masked, (0, 255, 0))][
                    [improperly_masked, not_masked, properly_masked].index(max([
                        improperly_masked, not_masked, properly_masked]))]

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
    run_classifier()
