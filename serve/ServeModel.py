import os
import argparse
import tensorflow as tf

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-ver', '--version',
        type=int,
        default='None',
        help='File name for the pbx file')
    argparser.add_argument(
        '-sp', '--savepath',
        default='None',
        help='Path directory of checkpoint file')
    argparser.add_argument(
        '-mn', '--modelname',
        default='None',
        help='File name for the pbx file')
    args = argparser.parse_args()

SERVE_PATH = './serve/{}/{}'.format(args.modelname, args.version)

checkpoint = tf.train.latest_checkpoint(args.savepath)

tf.reset_default_graph()

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(checkpoint + '.meta')
    saver.restore(sess, checkpoint)
    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name('inputs/x:0')
    prediction = graph.get_tensor_by_name('accuracy/prediction:0')

    builder = tf.saved_model.builder.SavedModelBuilder(SERVE_PATH)

    # Create tensor info
    model_input = tf.saved_model.utils.build_tensor_info(inputs)
    model_output = tf.saved_model.utils.build_tensor_info(prediction)

    # Build signature definition
    prediction_signature = \
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': model_input},
            outputs={'scores': model_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    signature_def_key = \
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            signature_def_key:
                prediction_signature
        },
        main_op=tf.tables_initializer())

    builder.save()

print('Done exporting')
