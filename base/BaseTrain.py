import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, config, logger, data_loader):
        self.sess = sess
        self.model = model
        self.config = config
        self.logger = logger
        self.data_loader = data_loader
        # a. Global variable is a variable that can be
        #    shared across multiple device,
        # b. Local variable is a variable that can't be
        #    train or doesn't compute its gradient
        self.init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        cur_epoch_tensor = self.model.cur_epoch_tensor.eval(self.sess)
        num_epochs = self.config.num_epochs
        for cur_epoch in range(cur_epoch_tensor, num_epochs + 1, 1):
            self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError
