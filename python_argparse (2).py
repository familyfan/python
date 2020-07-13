# -*- coding: utf-8 -*-
import tensorflow as tf
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                      help='Summaries log directory')
# 调用parser.parse_known_args()解析命令行参数
FLAGS, unparsed = parser.parse_known_args() # FLAGS是一个结构体
print(FLAGS.data_dir)
"""
/tmp/tensorflow/mnist/input_data
"""

#flags = tf.app.flags
#FLAGS = flags.FLAGS
#flags.DEFINE_boolean("enable_colored_log", False, "Enable colored log",
#                    "The glob pattern of train TFRecords files")
#flags.DEFINE_string("validate_tfrecords_file",
#                    "./data/a8a/a8a_test.libsvm.tfrecords",
#                    "The glob pattern of validate TFRecords files")
#flags.DEFINE_integer("label_size", 2, "Number of label size")
#flags.DEFINE_float("learning_rate", 0.01, "The learning rate")
#
#def main():
#    # Get hyperparameters
#    if FLAGS.enable_colored_log:
#        import coloredlogs
#        coloredlogs.install()
#    logging.basicConfig(level=logging.INFO)
#    FEATURE_SIZE = FLAGS.feature_size
#    LABEL_SIZE = FLAGS.label_size
#    ...
#    return 0
#
#if __name__ == ‘__main__’:
#    main()
#tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

