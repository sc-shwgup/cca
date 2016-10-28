#! /usr/bin/env python
import pandas 
import csv
import tensorflow as tf
import glob
import numpy as np
import os
import time
import datetime
import load_data_resolution
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir_resolution", "", "Checkpoint directory from training run")
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
X,X_train,X_test,dummy_y_train,dummy_y_test,num_classes,encoder = load_data_resolution.load_data_resolution_and_labels()

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    X,X_train,X_test,dummy_y_train,dummy_y_test,num_classes,encoder = load_data_resolution.load_data_resolution_and_labels()
    y_test = np.argmax(y_test, axis=1)
else:
    df = pandas.read_csv("./resolutionTest.csv")
#    df = pandas.read_csv("./sampleResolutionTest.csv")
    df = pandas.DataFrame(df)
    x_raw = df['Consumer complaint narrative'].astype('str')
    y_encoded_test = list(df['Resolution'].astype('str'))
 
y_test=[]
for i in y_encoded_test:
   y_test.append(encoder[i])
y_test = np.array(y_test)
# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir_resolution, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir_resolution)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = load_data_resolution.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
print("Predictions\n")

'''
with open('./encoder_resolution.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    encoder = dict(reader)

decoder = {float(v): k for k, v in encoder.items()}
# Output Visualization
predictions = pandas.DataFrame(all_predictions)
X_test = pandas.DataFrame(X_test)

y_test=[]
y_predicted=[]
for i in dummy_y_test:
    y_test.append(decoder[i])
y_test = pandas.DataFrame(y_test)

for i in predictions:
    y_predicted.append(decoder[i])
y_predicted = pandas.DataFrame(y_predicted)

output = pandas.concat([X_test,y_test,y_predcited],axis=1)

output.to_csv("./predictions.csv")
print("length prediction",len(all_predictions))
print("length y_test",len(y_test))
# Print accuracy if y_test is defined
'''
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
