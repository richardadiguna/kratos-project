import tensorflow as tf
import math


def augment(images, labels=[],
            vertical_flip=False,
            horizontal_flip=False,
            rotate=0):

    if images.dtype != tf.float32.name:
        shp = images.shape
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        images = tf.subtract(images, 0.5)
        images = tf.multiply(images, 2.0)

    with tf.name_scope('augmentation'):
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)

        if horizontal_flip:
            # coin is an array contains N number of boolean
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)

            # flip_transform
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(
                    coin,
                    tf.tile(
                        tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                    tf.tile(
                        tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
            # coin is an array contains N number of boolean
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)

            # flip_transform
            flip_transform = tf.convert_to_tensor(
                [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
                tf.where(
                    coin,
                    tf.tile(
                        tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                    tf.tile(
                        tf.expand_dims(identity, 0), [batch_size, 1])))

        if rotate > 0:
            # Set the angle radius
            angle_rad = rotate / 180 * math.pi
            angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
            transforms.append(
              tf.contrib.image.angles_to_projective_transforms(
                  angles, height, width))

        if transforms:
            images = tf.contrib.image.transform(
                images,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR')

    return images, labels
