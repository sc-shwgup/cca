#! /usr/bin/env python
import json
import tensorflow as tf
import glob
import numpy as np
import os
import sys
import time
import datetime
import load_data_issue
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
# Parameters
# ==================================================

def eval(x_raw,checkpoint_dir_issue):
# Eval Parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_string("checkpoint_dir_issue", "", "Checkpoint directory from training run")
    tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

# CHANGE THIS: Load data. Load your own data here
    if FLAGS.eval_train:
        x_raw, y_test = data_helpers.load_data_and_labels()
        y_test = np.argmax(y_test, axis=1)
    else:
         y_test = None     
# Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir_issue, "..", "vocab")
    print("vocab...")
    print(vocab_path)
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    print("\nEvaluating...\n")

# Evaluation
# ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir_issue)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
        # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        #saver = tf.train.Saver()
            saver.restore(sess,checkpoint_file)

        # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
            batches = load_data_issue.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
    print("predictions...",all_predictions[0])


    x_text, x_train,x_dev,y_train,y_dev,num_classes,encoder  = load_data_issue.load_data_issue_and_labels()

    decoder = {v: k for k, v in encoder.items()}
 
    data = {}
    data['Issue'] = decoder[all_predictions[0]]
    return data

