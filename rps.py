import utils
import cv2

def main():
    model = utils.load_model("data_model")
    cap = cv2.VideoCapture(0)

    utils.create_window("Display", (900, 600))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        resized = utils.resize_raw(frame)

        (name, probability) = utils.classify(model, resized)
        img = utils.label_image(name, probability, resized)

        cv2.imshow("Display", img)

        if utils.is_escape(cv2.waitKey(5)):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
