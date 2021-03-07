import sys
import pathlib

import cv2

import utils

def create_previews(data_gen, data_dir):
    options = dict(utils.load_options, batch_size=1)
    source = data_gen.flow_from_directory(
                data_dir, subset="training", **options)

    utils.create_window("Preview", (900, 600))

    for batch in source:
        preview_image = batch[0][0]
        preview_mat = utils.keras_img_to_opencv_mat(preview_image)
        cv2.imshow("Preview", preview_mat)

        key_code = cv2.waitKey(0)
        if utils.is_escape(key_code):
            break

    cv2.destroyAllWindows()

def main(data):
    data_gen = utils.prepare_data()

    data_folder = pathlib.Path(data)
    create_previews(data_gen, str(data_folder))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please supply data directory as argument")
        exit(-1)

    main(sys.argv[1])
