//
//  simplebaselineKeypoint.h
//  Tensorflow-lite
//
//  Created by Xiaobin Zhang on 2021/09/17.
//
//

#ifndef SIMPLEBASELINE_KEYPOINT_H
#define SIMPLEBASELINE_KEYPOINT_H

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

namespace simplebaselineKeypoint {

struct Settings {
  bool verbose = false;
  bool accel = false;
  bool input_floating = false;
  bool allow_fp16 = false;
  int loop_count = 1;
  float input_mean = 127.5f;
  float input_std = 127.5f;
  std::string model_name = "./model.tflite";
  tflite::FlatBufferModel* model;
  std::string input_img_name = "./dog.jpg";
  std::string classes_file_name = "./classes.txt";
  std::string input_layer_type = "uint8_t";
  int number_of_threads = 4;
  int number_of_warmup_runs = 2;
};

}  // namespace simplebaselineKeypoint

#endif  // SIMPLEBASELINE_KEYPOINT_H
