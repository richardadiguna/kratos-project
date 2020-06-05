import tensorflow as tf

from dataloader.DataLoader import DataLoader
from model.VGG16 import VGG16
from trainer.Trainer import Trainer
from utils.Parser import process_config
from utils.Logger import Logger
from utils.Utils import create_dirs, get_args


def main():
    args = get_args()
    m_config = process_config(args.config)

    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # create_dirs([config.summary_dir, config.checkpoint_dir])
        data = DataLoader(config=m_config)
        model = VGG16(data_loader=data, config=m_config)
        logger = Logger(sess=sess, config=m_config)

        trainer = Trainer(
            sess=sess,
            model=model,
            data=data,
            config=m_config,
            logger=logger)

        trainer.train()


if __name__ == '__main__':
    main()
