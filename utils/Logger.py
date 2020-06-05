import tensorflow as tf
import os


class Logger:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = tf.summary.FileWriter(
            os.path.join(self.config.summary_dir, "train"))
        self.test_summary_writer = tf.summary.FileWriter(
            os.path.join(self.config.summary_dir, "test"))

    def summarize(self, step, summarizer="train",
                  scope="", summaries_dict=None):

        train_sw = self.train_summary_writer
        test_sw = self.test_summary_writer

        summary_writer = train_sw if summarizer == "train" else test_sw

        if summaries_dict is not None:
            summary_list = []
            for tag, value in summaries_dict.items():
                if tag not in self.summary_ops:
                    if len(value.shape) <= 1:
                        self.summary_placeholders[tag] = tf.placeholder(
                            'float32', value.shape, name=tag)
                        self.summary_ops[tag] = tf.summary.scalar(
                            tag, self.summary_placeholders[tag])
                    else:
                        self.summary_placeholders[tag] = tf.placeholder(
                            'float32', [None] + list(value.shape[1:]),
                            name=tag)
                        self.summary_ops[tag] = tf.summary.image(
                            tag, self.summary_placeholders[tag])

                summary_list.append(
                    self.sess.run(
                        self.summary_ops[tag],
                        feed_dict={self.summary_placeholders[tag]: value}))

            for summary in summary_list:
                summary_writer.add_summary(summary, step)
