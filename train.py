#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import time

import tensorflow.keras.backend as K
#from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import TensorBoard, TerminateOnNaN

from simple_baselines.model import get_simple_baselines_model
from simple_baselines.data import keypoints_dataset
from simple_baselines.loss import get_loss
from simple_baselines.callbacks import EvalCallBack, CheckpointCleanCallBack
from common.utils import get_classes, get_matchpoints, optimize_tf_gpu
from common.model_utils import get_optimizer

# Try to enable Auto Mixed Precision on TF 2.0
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
optimize_tf_gpu(tf, K)


def main(args):
    log_dir = 'logs/000'
    os.makedirs(log_dir, exist_ok=True)

    class_names = get_classes(args.classes_path)
    num_classes = len(class_names)
    if args.matchpoint_path:
        matchpoints = get_matchpoints(args.matchpoint_path)
    else:
        matchpoints = None

    if args.mixed_precision:
        tf_major_version = float(tf.__version__[:3])
        if tf_major_version >= 2.1:
            # apply mixed_precision for valid TF version
            from tensorflow.keras.mixed_precision import experimental as mixed_precision

            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        else:
            raise ValueError('Tensorflow {} does not support mixed precision'.format(tf.__version__))


    # callbacks for training process
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    eval_callback = EvalCallBack(log_dir, args.dataset_path, class_names, args.model_input_shape, args.model_type)
    checkpoint_clean = CheckpointCleanCallBack(log_dir, max_val_keep=5)
    terminate_on_nan = TerminateOnNaN()

    callbacks = [tensorboard, eval_callback, terminate_on_nan, checkpoint_clean]

    # get train/val dataset
    train_dataset = keypoints_dataset(args.dataset_path, class_names,
                                input_shape=args.model_input_shape, is_train=True, matchpoints=matchpoints)
    val_dataset = keypoints_dataset(args.dataset_path, class_names,
                              input_shape=args.model_input_shape, is_train=False, matchpoints=None)

    num_train = train_dataset.get_dataset_size()
    num_val = val_dataset.get_dataset_size()

    train_gen = train_dataset.generator(args.batch_size, with_meta=False)

    # prepare loss function
    loss_func = get_loss(args.loss_type)

    # prepare optimizer
    optimizer = get_optimizer(args.optimizer, args.learning_rate, decay_type=None)
    #optimizer = RMSprop(lr=5e-4)

    # support multi-gpu training
    if args.gpu_num >= 2:
        # devices_list=["/gpu:0", "/gpu:1"]
        devices_list=["/gpu:{}".format(n) for n in range(args.gpu_num)]
        strategy = tf.distribute.MirroredStrategy(devices=devices_list)
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            # get multi-gpu train model, doesn't specify input size
            model = get_simple_baselines_model(args.model_type, num_classes, model_input_shape=None, freeze_level=args.freeze_level, weights_path=args.weights_path)
            # compile model
            model.compile(optimizer=optimizer, loss=loss_func)
    else:
        # get normal train model, doesn't specify input size
        model = get_simple_baselines_model(args.model_type, num_classes, model_input_shape=None, freeze_level=args.freeze_level, weights_path=args.weights_path)
        # compile model
        model.compile(optimizer=optimizer, loss=loss_func)

    print('Create {} Simple Baseline model with {} keypoints. train input size {}.'.format(args.model_type, num_classes, args.model_input_shape))
    model.summary()

    # Transfer training some epochs with frozen layers first if needed, to get a stable loss.
    initial_epoch = args.init_epoch
    epochs = initial_epoch + args.transfer_epoch
    print("Transfer training stage")
    print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(num_train, num_val, args.batch_size, args.model_input_shape))
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=num_train // args.batch_size,
                        epochs=epochs,
                        initial_epoch=initial_epoch,
                        verbose=1,
                        workers=1,
                        use_multiprocessing=False,
                        max_queue_size=10,
                        callbacks=callbacks)

    # Wait 2 seconds for next stage
    time.sleep(2)

    if args.decay_type:
        # rebuild optimizer to apply learning rate decay,
        # only after unfreeze all layers
        steps_per_epoch = max(1, num_train//args.batch_size)
        decay_steps = steps_per_epoch * (args.total_epoch - args.init_epoch - args.transfer_epoch)
        optimizer = get_optimizer(args.optimizer, args.learning_rate, decay_type=args.decay_type, decay_steps=decay_steps)


    # Unfreeze the whole network for further tuning
    # NOTE: more GPU memory is required after unfreezing the body
    print("Unfreeze and continue training, to fine-tune.")
    if args.gpu_num >= 2:
        with strategy.scope():
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=optimizer, loss=loss_func) # recompile to apply the change
    else:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=optimizer, loss=loss_func) # recompile to apply the change

    print('Train on {} samples, val on {} samples, with batch size {}, input_size {}.'.format(num_train, num_val, args.batch_size, args.model_input_shape))
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=num_train // args.batch_size,
                        epochs=args.total_epoch,
                        initial_epoch=epochs,
                        verbose=1,
                        workers=1,
                        use_multiprocessing=False,
                        max_queue_size=10,
                        callbacks=callbacks)

    # Finally store model
    model.save(os.path.join(log_dir, 'trained_final.h5'))
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model definition options
    parser.add_argument('--model_type', type=str, required=False, default='resnet50_deconv',
        help='Simple Baselines model type: resnet50_deconv/resnet50_upsample/mobilenetv2_upsample_lite/, default=%(default)s')
    parser.add_argument('--model_input_shape', type=str, required=False, default='256x256',
        help = "model image input shape as <height>x<width>, default=%(default)s")
    parser.add_argument('--weights_path', type=str, required=False, default=None,
        help = "Pretrained model/weights file for fine tune")

    # Data options
    parser.add_argument('--dataset_path', type=str, required=False, default='data/mpii',
        help='dataset path containing images and annotation file, default=%(default)s')
    parser.add_argument('--classes_path', type=str, required=False, default='configs/mpii_classes.txt',
        help='path to keypoint class definitions, default=%(default)s')
    parser.add_argument('--matchpoint_path', type=str, required=False, default='configs/mpii_match_point.txt',
        help='path to matching keypoint definitions for horizontal/vertical flipping image, default=%(default)s')

    # Training options
    parser.add_argument("--batch_size", type=int, required=False, default=16,
        help='batch size for training, default=%(default)s')
    parser.add_argument('--optimizer', type=str, required=False, default='rmsprop',
        help = "optimizer for training (adam/rmsprop/sgd), default=%(default)s")
    parser.add_argument('--loss_type', type=str, required=False, default='mse', choices=['mse', 'mae', 'weighted_mse', 'smooth_l1', 'huber'],
        help = "loss type for training (mse/mae/weighted_mse/smooth_l1/huber), default=%(default)s")
    parser.add_argument('--learning_rate', type=float, required=False, default=5e-4,
        help = "Initial learning rate, default=%(default)s")
    parser.add_argument('--decay_type', type=str, required=False, default=None, choices=[None, 'cosine', 'exponential', 'polynomial', 'piecewise_constant'],
        help = "Learning rate decay type, default=%(default)s")
    parser.add_argument('--mixed_precision', default=False, action="store_true",
        help='Use mixed precision mode in training, only for TF>2.1')

    parser.add_argument('--transfer_epoch', type=int, required=False, default=1,
        help = "Transfer training stage epochs, default=%(default)s")
    parser.add_argument('--freeze_level', type=int, required=False, default=1, choices=[0, 1, 2],
        help = "Freeze level of the model in transfer training stage. 0:NA/1:backbone/2:only open prediction layer, default=%(default)s")
    parser.add_argument("--init_epoch", type=int, required=False, default=0,
        help="initial training epochs for fine tune training, default=%(default)s")
    parser.add_argument("--total_epoch", type=int, required=False, default=100,
        help="total training epochs, default=%(default)s")
    parser.add_argument('--gpu_num', type=int, required=False, default=1,
        help='Number of GPU to use, default=%(default)s')

    args = parser.parse_args()
    height, width = args.model_input_shape.split('x')
    args.model_input_shape = (int(height), int(width))

    main(args)
