#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import cv2
from PIL import Image
import numpy as np
import time
from timeit import default_timer as timer
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K

from simple_baselines.model import get_simple_baselines_model
from simple_baselines.data import OUTPUT_STRIDE
from simple_baselines.postprocess import post_process_heatmap, post_process_heatmap_simple
from common.data_utils import preprocess_image
from common.utils import get_classes, get_skeleton, render_skeleton, optimize_tf_gpu

from detector import detect_person, get_anchors, get_square_box

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

default_config = {
        "model_type": 'resnet50_deconv',
        "model_input_shape": (256, 256),
        "conf_threshold": 0.1,
        "classes_path": os.path.join('configs', 'mpii_classes.txt'),
        "skeleton_path": None,
        "weights_path": os.path.join('weights', 'model.h5'),

        # YOLOv3 person detection model info
        "det_model_path": os.path.join('detector', 'yolo3_mobilenet_lite_320_coco.h5'),
        "det_anchors_path": os.path.join('detector', 'yolo3_anchors.txt'),
        "det_classes_path": os.path.join('detector', 'coco_classes.txt'),
        "det_model_input_shape": (320, 320),

        "gpu_num" : 1,
    }


class SimpleBaselines(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        super(SimpleBaselines, self).__init__()
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        if self.skeleton_path:
            self.skeleton_lines = get_skeleton(self.skeleton_path)
        else:
            self.skeleton_lines = None
        self.class_names = get_classes(self.classes_path)
        self.model = self._generate_model()

        self._init_detection_model()
        K.set_learning_phase(0)

    def _generate_model(self):
        '''to generate the bounding boxes'''
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_classes = len(self.class_names)

        # construct model and load weights.
        model = get_simple_baselines_model(self.model_type, num_classes, model_input_shape=self.model_input_shape)
        model.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        model.summary()
        return model

    def _init_detection_model(self):
        self.det_anchors = get_anchors(self.det_anchors_path)
        self.det_class_names = get_classes(self.det_classes_path)

        #custom_object_dict = get_custom_objects()
        #model = load_model(model_path, compile=False, custom_objects=custom_object_dict)
        self.det_model = load_model(self.det_model_path, compile=False)
        return

    def detect_image(self, image):
        image_array = np.array(image, dtype='uint8')

        start = time.time()
        person_boxes, person_scores = detect_person(image, self.det_model, self.det_anchors, self.det_class_names, self.det_model_input_shape)
        for box, score in zip(person_boxes, person_scores):
            raw_xmin, raw_ymin, raw_xmax, raw_ymax = map(int, box)

            # expand person bbox to square
            xmin, ymin, xmax, ymax = get_square_box(box, image.size)

            # crop person image area as keypoint model input
            person_image = Image.fromarray(image_array[ymin:ymax, xmin:xmax])
            person_array = np.array(person_image, dtype='uint8')

            image_data = preprocess_image(person_image, self.model_input_shape)

            # NOTE: image_size and scale in (w,h) format, but
            #       self.model_input_shape in (h,w) format
            image_size = person_image.size
            scale = (image_size[0] * 1.0 / self.model_input_shape[1], image_size[1] * 1.0 / self.model_input_shape[0])

            keypoints = self.predict(image_data)

            # rescale keypoints back to origin image size
            keypoints_dict = dict()
            for i, keypoint in enumerate(keypoints):
                keypoints_dict[self.class_names[i]] = (keypoint[0] * scale[0] * OUTPUT_STRIDE + xmin, keypoint[1] * scale[1] * OUTPUT_STRIDE + ymin, keypoint[2])

            # draw bbox rectangle on image
            cv2.rectangle(image_array, (raw_xmin, raw_ymin), (raw_xmax, raw_ymax), (255, 0, 0), 1, cv2.LINE_AA)
            #cv2.rectangle(image_array, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1, cv2.LINE_AA)

            # draw keypoint skeleton on image
            image_array = render_skeleton(image_array, keypoints_dict, self.skeleton_lines, self.conf_threshold)

        end = time.time()
        print("Inference time: {:.8f}s".format(end - start))
        return Image.fromarray(image_array)


    def detect_image_batch(self, image):
        """
        run batch inference on keypoint model for multi person inputs
        """
        image_array = np.array(image, dtype='uint8')
        start = time.time()

        person_boxes, person_scores = detect_person(image, self.det_model, self.det_anchors, self.det_class_names, self.det_model_input_shape)

        batch_image_data = []
        batch_scale = []
        batch_raw_box = []
        batch_box = []
        for box, score in zip(person_boxes, person_scores):
            raw_xmin, raw_ymin, raw_xmax, raw_ymax = map(int, box)

            # expand person bbox to square
            xmin, ymin, xmax, ymax = get_square_box(box, image.size)

            # crop person image area as keypoint model input
            person_image = Image.fromarray(image_array[ymin:ymax, xmin:xmax])
            person_array = np.array(person_image, dtype='uint8')

            image_data = preprocess_image(person_image, self.model_input_shape)

            # NOTE: image_size and scale in (w,h) format, but
            #       self.model_input_shape in (h,w) format
            image_size = person_image.size
            scale = (image_size[0] * 1.0 / self.model_input_shape[1], image_size[1] * 1.0 / self.model_input_shape[0])

            # merge batched info for inference
            batch_scale.append(scale)
            batch_raw_box.append((raw_xmin, raw_ymin, raw_xmax, raw_ymax))
            batch_box.append((xmin, ymin, xmax, ymax))
            # here we strip the batch dim in image_data before append
            batch_image_data.append(image_data[0])

        # just return if no person
        if len(batch_image_data) == 0:
            return Image.fromarray(image_array)

        # run batch inference
        batch_image_data = np.array(batch_image_data)
        batch_keypoints = self.batch_predict(batch_image_data)

        # handle batch inference result
        for i, keypoints in enumerate(batch_keypoints):
            scale = batch_scale[i]
            raw_xmin, raw_ymin, raw_xmax, raw_ymax = batch_raw_box[i]
            xmin, ymin, xmax, ymax = batch_box[i]

            # rescale keypoints back to origin image size
            keypoints_dict = dict()
            for j, keypoint in enumerate(keypoints):
                keypoints_dict[self.class_names[j]] = (keypoint[0] * scale[0] * OUTPUT_STRIDE + xmin, keypoint[1] * scale[1] * OUTPUT_STRIDE + ymin, keypoint[2])

            # draw bbox rectangle on image
            cv2.rectangle(image_array, (raw_xmin, raw_ymin), (raw_xmax, raw_ymax), (255, 0, 0), 1, cv2.LINE_AA)
            #cv2.rectangle(image_array, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1, cv2.LINE_AA)

            # draw the keypoint skeleton on image
            image_array = render_skeleton(image_array, keypoints_dict, self.skeleton_lines, self.conf_threshold)

        end = time.time()
        print("Inference time: {:.8f}s".format(end - start))
        return Image.fromarray(image_array)


    def predict(self, image_data):
        # get final predict heatmap
        prediction = self.model.predict(image_data)
        if isinstance(prediction, list):
            prediction = prediction[-1]
        heatmap = prediction[0]

        # parse out predicted keypoint from heatmap
        keypoints = post_process_heatmap_simple(heatmap)

        return keypoints


    def batch_predict(self, image_data):
        # get batch predict heatmap
        prediction = self.model.predict_on_batch(image_data)
        if isinstance(prediction, list):
            prediction = prediction[-1]

        batch_keypoints = []
        for heatmap in prediction:
            # parse out predicted keypoint from heatmap
            keypoints = post_process_heatmap_simple(heatmap)
            batch_keypoints.append(keypoints)

        return batch_keypoints

    def dump_model_file(self, output_model_file):
        self.model.save(output_model_file)


def detect_video(simple_baselines, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(0 if video_path == '0' else video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = cv2.VideoWriter_fourcc(*'XVID') if video_path == '0' else int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, (5. if video_path == '0' else video_fps), video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = simple_baselines.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release everything if job is finished
    vid.release()
    if isOutput:
        out.release()
    cv2.destroyAllWindows()



def detect_img(simple_baselines):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img).convert('RGB')
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = simple_baselines.detect_image(image)
            r_image.show()



if __name__ == "__main__":
    # class SimpleBaselines defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='demo or dump out Simple Baselines h5 model')
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_type', type=str,
        help='model type, default ' + str(SimpleBaselines.get_defaults("model_type"))
    )
    parser.add_argument(
        '--model_input_shape', type=str,
        help='model image input shape as <height>x<width>, default ' +
        str(SimpleBaselines.get_defaults("model_input_shape")[0])+'x'+str(SimpleBaselines.get_defaults("model_input_shape")[1]),
        default=str(SimpleBaselines.get_defaults("model_input_shape")[0])+'x'+str(SimpleBaselines.get_defaults("model_input_shape")[1])
    )
    parser.add_argument(
        '--weights_path', type=str,
        help='path to model weight file, default ' + SimpleBaselines.get_defaults("weights_path")
    )
    parser.add_argument(
        '--classes_path', type=str, required=False,
        help='path to keypoint class definitions, default ' + SimpleBaselines.get_defaults("classes_path")
    )
    parser.add_argument(
        '--skeleton_path', type=str, required=False,
        help='path to keypoint skeleton definitions, default ' + str(SimpleBaselines.get_defaults("skeleton_path"))
    )
    parser.add_argument(
        '--conf_threshold', type=float,
        help='confidence threshold, default ' + str(SimpleBaselines.get_defaults("conf_threshold"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )
    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    '''
    Command line positional arguments -- for model dump
    '''
    parser.add_argument(
        '--dump_model', default=False, action="store_true",
        help='Dump out training model to inference model'
    )

    parser.add_argument(
        '--output_model_file', type=str,
        help='output inference model file'
    )

    args = parser.parse_args()
    # param parse
    if args.model_input_shape:
        height, width = args.model_input_shape.split('x')
        args.model_input_shape = (int(height), int(width))

    # get wrapped inference object
    simple_baselines = SimpleBaselines(**vars(args))

    if args.dump_model:
        """
        Dump out training model to inference model
        """
        if not args.output_model_file:
            raise ValueError('output model file is not specified')

        print('Dumping out training model to inference model')
        simple_baselines.dump_model_file(args.output_model_file)
        sys.exit()

    if args.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in args:
            print(" Ignoring remaining command line arguments: " + args.input + "," + args.output)
        detect_img(simple_baselines)
    elif "input" in args:
        detect_video(simple_baselines, args.input, args.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")

