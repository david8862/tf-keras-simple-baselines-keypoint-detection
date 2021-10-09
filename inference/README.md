## C++ on-device (X86/ARM) inference app for Simple Baselines Model

Here are some C++ implementation of the on-device inference for trained Simple Baselines models, including forward propagation of the model, heatmap postprocess and coordinate rescale. Now we have 2 approaches with different inference engine for that:

* Tensorflow-Lite (verified on commit id: 1b8f5bc8011a1e85d7a110125c852a4f431d0f59)
* [MNN](https://github.com/alibaba/MNN) from Alibaba (verified on release: [1.0.0](https://github.com/alibaba/MNN/releases/tag/1.0.0))


### MNN

1. Install Python runtime and Build libMNN

Refer to [MNN build guide](https://www.yuque.com/mnn/cn/build_linux), we need to prepare cmake & protobuf first for MNN build. And since MNN support both X86 & ARM platform, we can do either native compile or ARM cross-compile
```
# apt install cmake autoconf automake libtool ocl-icd-opencl-dev
# wget https://github.com/google/protobuf/releases/download/v3.4.1/protobuf-cpp-3.4.1.tar.gz
# tar xzvf protobuf-cpp-3.4.1.tar.gz
# cd protobuf-3.4.1
# ./autogen.sh
# ./configure && make && make check && make install && ldconfig
# pip install --upgrade pip && pip install --upgrade mnn

# git clone https://github.com/alibaba/MNN.git <Path_to_MNN>
# cd <Path_to_MNN>
# ./schema/generate.sh
# ./tools/script/get_model.sh  # optional
# mkdir build && cd build
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>]
        [-DMNN_BUILD_QUANTOOLS=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TRAIN=ON -MNN_BUILD_TRAIN_MINI=ON -MNN_USE_OPENCV=OFF] ..
        && make -j4

### MNN OpenCL backend build
# apt install ocl-icd-opencl-dev
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>]
        [-DMNN_OPENCL=ON -DMNN_SEP_BUILD=OFF -DMNN_USE_SYSTEM_LIB=ON] ..
        && make -j4
```
If you want to do cross compile for ARM platform, "CMAKE_TOOLCHAIN_FILE" should be specified

"MNN_BUILD_QUANTOOLS" is for enabling MNN Quantization tool

"MNN_BUILD_CONVERTER" is for enabling MNN model converter

"MNN_BUILD_BENCHMARK" is for enabling on-device inference benchmark tool

"MNN_BUILD_TRAIN" related are for enabling MNN training tools


2. Build demo inference application
```
# cd tf-keras-simple-baselines-keypoint-detection/inference/MNN
# mkdir build && cd build
# cmake -DMNN_ROOT_PATH=<Path_to_MNN> [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
# make
```

3. Convert trained Simple Baselines model to MNN model

Refer to [Model dump](https://github.com/david8862/tf-keras-simple-baselines-keypoint-detection#model-dump), [Tensorflow model convert](https://github.com/david8862/tf-keras-simple-baselines-keypoint-detection#tensorflow-model-convert) and [MNN model convert](https://www.yuque.com/mnn/cn/model_convert), we need to:

* dump out inference model from training checkpoint:

    ```
    # python demo.py --model_type=resnet50_deconv --model_input_shape=256x256 --weights_path=logs/<checkpoint>.h5 --classes_path=configs/mpii_classes.txt --dump_model --output_model_file=model.h5
    ```

* convert keras .h5 model to tensorflow frozen pb model:

    ```
    # python keras_to_tensorflow.py
        --input_model="path/to/keras/model.h5"
        --output_model="path/to/save/model.pb"
    ```

* convert TF pb model to MNN model:

    ```
    # cd <Path_to_MNN>/build/
    # ./MNNConvert -f TF --modelFile model.pb --MNNModel model.pb.mnn --bizCode MNN
    ```
    or

    ```
    # mnnconvert -f TF --modelFile model.pb --MNNModel model.pb.mnn
    ```

MNN support Post Training Integer quantization, so we can use its python CLI interface to do quantization on the generated .mnn model to get quantized .mnn model for ARM acceleration . A json config file [quantizeConfig.json](https://github.com/david8862/tf-keras-simple-baselines-keypoint-detection/blob/master/inference/MNN/configs/quantizeConfig.json) is needed to describe the feeding data:

* Quantized MNN model:

    ```
    # cd <Path_to_MNN>/build/
    # ./quantized.out model.pb.mnn model_quant.pb.mnn quantizeConfig.json
    ```
    or

    ```
    # mnnquant model.pb.mnn model_quant.pb.mnn quantizeConfig.json
    ```

4. Run validate script to check MNN model
```
# cd tf-keras-simple-baselines-keypoint-detection/tools/evaluation/
# python validate_simple_baselines.py --model_path=model_quant.pb.mnn --classes_path=../../configs/mpii_classes.txt --skeleton_path=../../configs/mpii_skeleton.txt --image_file=../../example/fitness.jpg --model_input_shape=256x256 --loop_count=5
```

Visualized detection result:

<p align="center">
  <img src="../assets/dog_inference.jpg">
</p>

#### You can also use [eval.py](https://github.com/david8862/tf-keras-simple-baselines-keypoint-detection#evaluation) to do evaluation on the MNN model


5. Run application to do inference with model, or put all the assets to your ARM board and run if you use cross-compile
```
# cd tf-keras-simple-baselines-keypoint-detection/inference/MNN/build
# ./simplebaselineKeypoint -h
Usage: simplebaselineKeypoint
--mnn_model, -m: model_name.mnn
--image, -i: image_name.jpg
--classes, -l: classes labels for the model
--input_mean, -b: input mean
--input_std, -s: input standard deviation
--threads, -t: number of threads
--count, -c: loop model run for certain times
--warmup_runs, -w: number of warmup runs

# ./simplebaselineKeypoint -m model.pb.mnn -i ../../../example/fitness.jpg -l ../../../configs/mpii_classes.txt -t 8 -c 10 -w 3
image_input: name:image_input, width:256, height:256, channel:3, dim_type:TENSORFLOW
num_outputs: 1
num_classes: 16
origin image size: width:640, height:640, channel:3
model invoke average time: 18.475500 ms
output tensor name: heatmap_predict/BiasAdd
output tensor shape: width:64 , height:64, channel: 16
heatmap shape: batch:1, width:64 , height:64, channel: 16
Caffe format: NCHW
batch 0:
heatmap_postprocess time: 0.122000 ms
prediction_list length: 16
Keypoint Detection result:
right_ankle 0.712625 (270, 580)
right_knee 0.664650 (280, 450)
right_hip 0.553376 (280, 300)
left_hip 0.577002 (350, 300)
left_knee 0.737033 (380, 440)
left_ankle 0.693533 (350, 560)
plevis 0.664985 (320, 300)
thorax 0.726077 (310, 140)
upper_neck 0.788715 (310, 110)
head_top 0.725502 (300, 30)
right_wrist 0.706904 (230, 300)
right_elbow 0.717830 (240, 230)
right_shoulder 0.699679 (250, 150)
left_shoulder 0.660681 (380, 140)
left_elbow 0.766712 (410, 220)
left_wrist 0.742856 (420, 300)
```
Here the [classes](https://github.com/david8862/tf-keras-simple-baselines-keypoint-detection/blob/master/configs/mpii_classes.txt) file format are the same as used in training part




### Tensorflow-Lite

1. Build TF-Lite lib

We can do either native compile for X86 or cross-compile for ARM

```
# git clone https://github.com/tensorflow/tensorflow <Path_to_TF>
# cd <Path_to_TF>
# ./tensorflow/lite/tools/make/download_dependencies.sh
# make -f tensorflow/lite/tools/make/Makefile   #for X86 native compile
# ./tensorflow/lite/tools/make/build_rpi_lib.sh #for ARM cross compile, e.g Rasperberry Pi
```

2. Build demo inference application
```
# cd tf-keras-simple-baselines-keypoint-detection/inference/tflite
# mkdir build && cd build
# cmake -DTF_ROOT_PATH=<Path_to_TF> [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] [-DTARGET_PLAT=<target>] ..
# make
```
If you want to do cross compile for ARM platform, "CMAKE_TOOLCHAIN_FILE" and "TARGET_PLAT" should be specified. Refer [CMakeLists.txt](https://github.com/david8862/tf-keras-simple-baselines-keypoint-detection/blob/master/inference/tflite/CMakeLists.txt) for details.

3. Convert trained Simple Baselines model to tflite model

Currently, We only support dumping out keras .h5 model to Float32 .tflite model:

* dump out inference model from training checkpoint:

    ```
    # python demo.py --model_type=resnet50_deconv --model_input_shape=256x256 --weights_path=logs/<checkpoint>.h5 --classes_path=configs/mpii_classes.txt --dump_model --output_model_file=model.h5
    ```

* convert keras .h5 model to Float32 tflite model:

    ```
    # tflite_convert --keras_model_file=model.h5 --output_file=model.tflite
    ```

4. Run validate script to check TFLite model
```
# cd tf-keras-simple-baselines-keypoint-detection/tools/evaluation/
# python validate_simple_baselines.py --model_path=model.tflite --classes_path=../../configs/mpii_classes.txt --skeleton_path=../../configs/mpii_skeleton.txt --image_file=../../example/fitness.jpg --model_input_shape=256x256 --loop_count=5
```

5. Run application to do inference with model, or put assets to ARM board and run if cross-compile
```
# cd tf-keras-simple-baselines-keypoint-detection/inference/tflite/build
# ./simplebaselineKeypoint -h
Usage: simplebaselineKeypoint
--tflite_model, -m: model_name.tflite
--image, -i: image_name.jpg
--classes, -l: classes labels for the model
--input_mean, -b: input mean
--input_std, -s: input standard deviation
--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not
--threads, -t: number of threads
--count, -c: loop interpreter->Invoke() for certain times
--warmup_runs, -w: number of warmup runs
--verbose, -v: [0|1] print more information

# ./simplebaselineKeypoint -m model.tflite -i ../../../example/fitness.jpg -l ../../../configs/mpii_classes.txt -t 8 -c 10 -w 3
resolved reporter
num_classes: 16
origin image size: width:640, height:640, channel:3
invoked average time:86.2127 ms
batch 0
heatmap_postprocess time: 0.114 ms
prediction_list length: 16
Keypoint Detection result:
right_ankle 0.677985 (270, 590)
right_knee 0.660247 (280, 450)
right_hip 0.525227 (280, 300)
left_hip 0.560349 (350, 300)
left_knee 0.700154 (380, 440)
left_ankle 0.639327 (350, 560)
plevis 0.65256 (320, 300)
thorax 0.712529 (310, 140)
upper_neck 0.761622 (310, 110)
head_top 0.700116 (300, 30)
right_wrist 0.704193 (230, 300)
right_elbow 0.763515 (240, 230)
right_shoulder 0.731346 (250, 140)
left_shoulder 0.63525 (380, 130)
left_elbow 0.765785 (410, 220)
left_wrist 0.729243 (420, 300)
```

### TODO
- [ ] support letterbox input image in validate script and C++ inference code

