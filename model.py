import tensorflow as tf
import tensorflow.contrib.slim as slim


class DTN(object):
    '''
    Domain Transfer Network
    '''

    def __init__(self, mode='train', learning_rate=0.0003,
                 n_classes=10, margin=2048.0, ucn_weight=5.0,
                 f_weight=3.0, reconst_weight=15.0):
        self.mode = mode
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.margin = margin
        self.ucn_weight = ucn_weight
        self.f_weight = f_weight
        self.reconst_weight = reconst_weight

    def content_extractor(self, images, reuse=False):
        n_classes = self.n_classes
        # images: (batch, 64, 64, 3) or (batch, 64, 64, 1)
        if images.get_shape()[3] == 1:
            # Replicate the gray scale image 3 times.
            images = tf.image.grayscale_to_rgb(images)

        with tf.variable_scope('content_extractor', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME',
                                activation_fn=None,
                                stride=2,
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95,
                                    center=True, scale=True,
                                    activation_fn=tf.nn.relu,
                                    is_training=self.mode in ['train',
                                                              'pretrain']):

                    # (batch_size, 32, 32, 32)
                    net = slim.conv2d(images, 32, [3, 3],
                                      scope='conv1')
                    net = slim.batch_norm(net, scope='bn1')
                    # (batch_size, 16, 16, 64)
                    net = slim.conv2d(net, 64, [3, 3],
                                      scope='conv2')
                    net = slim.batch_norm(net, scope='bn2')
                    # (batch_size, 8, 8, 128)
                    net = slim.conv2d(net, 128, [3, 3],
                                      scope='conv3')
                    net = slim.batch_norm(net, scope='bn3')
                    # (batch_size, 4, 4, 256)
                    net = slim.conv2d(net, 256, [3, 3],
                                      scope='conv4')
                    net = slim.batch_norm(net, scope='bn4')
                    # (batch_size, 1, 1, 512)
                    net = slim.conv2d(net, 512, [4, 4], padding='VALID',
                                      scope='conv5')
                    net = slim.batch_norm(net, activation_fn=tf.nn.tanh,
                                          scope='bn5')
                    if self.mode == 'pretrain':
                        # (batch_size, 1, 1, n_classes)
                        net = slim.conv2d(net, n_classes, [1, 1],
                                          padding='VALID',
                                          scope='out')
                        # (batch_size, n_classes)
                        net = slim.flatten(net)
                    return net

    def generator(self, inputs, reuse=False):
        # inputs: (batch, 1, 1, 512)
        with tf.variable_scope('generator', reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose],
                                padding='SAME', activation_fn=None,
                                stride=2,
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm],
                                    decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.relu,
                                    is_training=(self.mode == 'train')):

                    # (batch_size, 4, 4, 512)
                    net = slim.conv2d_transpose(
                        inputs, 512, [4, 4],
                        padding='VALID', scope='conv_transpose1')
                    net = slim.batch_norm(net, scope='bn1')
                    # (batch_size, 8, 8, 256)
                    net = slim.conv2d_transpose(net, 256, [3, 3],
                                                scope='conv_transpose2')
                    net = slim.batch_norm(net, scope='bn2')
                    # (batch_size, 16, 16, 128)
                    net = slim.conv2d_transpose(net, 128, [3, 3],
                                                scope='conv_transpose3')
                    net = slim.batch_norm(net, scope='bn3')
                    # (batch_size, 32, 32, 64)
                    net = slim.conv2d_transpose(net, 64, [3, 3],
                                                scope='conv_transpose4')
                    net = slim.batch_norm(net, scope='bn4')
                    # (batch_size, 64, 64, 3)
                    net = slim.conv2d_transpose(net, 3, [3, 3],
                                                activation_fn=tf.nn.tanh,
                                                scope='conv_transpose5')
                    return net

    def discriminator(self, images, reuse=False):
        # images: (batch, 64, 64, 3)
        with tf.variable_scope('discriminator', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME',
                                activation_fn=None,
                                stride=2,
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95,
                                    center=True, scale=True,
                                    activation_fn=tf.nn.relu,
                                    is_training=(self.mode == 'train')):

                    # (batch_size, 32, 32, 64)
                    net = slim.conv2d(images, 64, [3, 3],
                                      activation_fn=tf.nn.relu, scope='conv1')
                    net = slim.batch_norm(net, scope='bn1')
                    # (batch_size, 16, 16, 128)
                    net = slim.conv2d(net, 128, [3, 3], scope='conv2')
                    net = slim.batch_norm(net, scope='bn2')
                    # (batch_size, 8, 8, 256)
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3')
                    net = slim.batch_norm(net, scope='bn3')
                    # (batch_size, 4, 4, 512)
                    net = slim.conv2d(net, 512, [3, 3], scope='conv4')
                    net = slim.batch_norm(net, scope='bn4')
                    # (batch_size, 1, 1, 3)
                    net = slim.conv2d(net, 3, [4, 4], padding='VALID',
                                      scope='conv5')
                    net = slim.flatten(net)
                    return net

    def build_model(self):
        if self.mode == 'pretrain':
            self.images = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                         'real_faces')
            self.labels = tf.placeholder(tf.int64, [None], 'face_labels')
            self.pos_ones = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                           'positive_pair_one')
            self.pos_twos = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                           'positive_pair_two')
            self.neg_ones = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                           'negative_pair_one')
            self.neg_twos = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                           'negative_pair_two')

            # logits and accuracy
            self.f_pos1 = self.content_extractor(self.pos_ones)
            self.f_pos2 = self.content_extractor(self.pos_twos, reuse=True)
            self.f_neg1 = self.content_extractor(self.neg_ones, reuse=True)
            self.f_neg2 = self.content_extractor(self.neg_twos, reuse=True)
            self.logits = self.content_extractor(self.images, reuse=True)
            self.pred = tf.argmax(self.logits, 1)
            self.correct_pred = tf.equal(self.pred, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred,
                                                   tf.float32))

            # loss and train op
            self.loss_ucn_pos = tf.reduce_mean(
                tf.square(self.f_pos1 - self.f_pos2))
            neg_diff = tf.square(self.f_neg1 - self.f_neg2)
            self.loss_ucn_neg = tf.reduce_mean(
                tf.maximum(0., self.margin - neg_diff))
            self.loss_ucn = self.loss_ucn_pos + self.loss_ucn_neg
            self.loss_class = \
                tf.losses.sparse_softmax_cross_entropy(self.labels,
                                                       self.logits)
            self.loss = self.loss_class + self.ucn_weight * self.loss_ucn
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = slim.learning.create_train_op(self.loss,
                                                          self.optimizer,
                                                          clip_gradient_norm=1)

            # summary op
            loss_class_summary = tf.summary.scalar('classification_loss',
                                                   self.loss_class)
            loss_ucn_summary = tf.summary.scalar('ucn loss', self.loss_ucn)
            loss_ucn_pos_summary = tf.summary.scalar('ucn pos loss',
                                                     self.loss_ucn_pos)
            loss_ucn_neg_summary = tf.summary.scalar('ucn neg loss',
                                                     self.loss_ucn_neg)
            loss_summary = tf.summary.scalar('combined loss', self.loss)
            accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
            logits_summary = tf.summary.histogram('probability distribution',
                                                  tf.nn.softmax(self.logits))
            self.summary_op = tf.summary.merge([loss_summary,
                                                loss_class_summary,
                                                loss_ucn_summary,
                                                loss_ucn_pos_summary,
                                                loss_ucn_neg_summary,
                                                logits_summary,
                                                accuracy_summary])

        elif self.mode == 'eval':
            self.images = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                         'real_faces')

            # source domain
            self.fx = self.content_extractor(self.images)
            self.sampled_images = self.generator(self.fx)

        elif self.mode == 'train':
            self.src_images = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                             'real_faces')
            self.trg_images = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                             'caricature_faces')
            self.pos_ones = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                           'positive_pair_one')
            self.pos_twos = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                           'positive_pair_two')
            self.neg_ones = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                           'negative_pair_one')
            self.neg_twos = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                           'negative_pair_two')

            # features for labelled pairs
            self.f_pos1 = self.content_extractor(self.pos_ones)
            self.f_pos2 = self.content_extractor(self.pos_twos, reuse=True)
            self.f_neg1 = self.content_extractor(self.neg_ones, reuse=True)
            self.f_neg2 = self.content_extractor(self.neg_twos, reuse=True)

            # source domain
            self.fx = self.content_extractor(self.src_images, reuse=True)
            self.fake_images = self.generator(self.fx)
            self.logits = self.discriminator(self.fake_images)
            self.fgfx = self.content_extractor(self.fake_images, reuse=True)
            ones = tf.ones_like(self.logits)
            class_one = tf.multiply(ones, tf.constant([1, 0, 0],
                                                      dtype=tf.float32))
            class_two = tf.multiply(ones, tf.constant([0, 1, 0],
                                                      dtype=tf.float32))
            class_three = tf.multiply(ones, tf.constant([0, 0, 1],
                                                        dtype=tf.float32))

            # ucn loss
            self.loss_ucn_pos = tf.reduce_mean(
                tf.square(self.f_pos1 - self.f_pos2))
            neg_diff = tf.square(self.f_neg1 - self.f_neg2)
            self.loss_ucn_neg = tf.reduce_mean(
                tf.maximum(0., self.margin - neg_diff))
            self.loss_ucn = self.loss_ucn_pos + self.loss_ucn_neg

            # dtn loss src
            self.d_loss_src = tf.losses.softmax_cross_entropy(
                class_one, self.logits)            # L_D D1
            self.g_loss_src = tf.losses.softmax_cross_entropy(
                class_three, self.logits)          # L_GANG D3
            self.f_loss_src = self.f_weight * (
                tf.reduce_mean(tf.square(self.fx - self.fgfx))
                + self.loss_ucn * self.ucn_weight)

            # optimizer
            self.d_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.f_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)

            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            g_vars = [var for var in t_vars if 'generator' in var.name]
            f_vars = [var for var in t_vars if 'content_extractor' in var.name]

            # train op
            with tf.variable_scope('source_train_op', reuse=False):
                self.d_train_op_src = slim.learning.create_train_op(
                    self.d_loss_src,
                    self.d_optimizer_src,
                    variables_to_train=d_vars)
                self.g_train_op_src = slim.learning.create_train_op(
                    self.g_loss_src,
                    self.g_optimizer_src,
                    variables_to_train=g_vars)
                self.f_train_op_src = slim.learning.create_train_op(
                    self.f_loss_src,
                    self.f_optimizer_src,
                    variables_to_train=f_vars)

            # summary op
            d_loss_src_summary = tf.summary.scalar('src_d_loss',
                                                   self.d_loss_src)
            g_loss_src_summary = tf.summary.scalar('src_g_loss',
                                                   self.g_loss_src)
            f_loss_src_summary = tf.summary.scalar('src_f_loss',
                                                   self.f_loss_src)
            origin_images_summary = tf.summary.image('src_origin_images',
                                                     self.src_images)
            sampled_images_summary = tf.summary.image('src_sampled_images',
                                                      self.fake_images)
            loss_ucn_summary = tf.summary.scalar('ucn loss',
                                                 self.loss_ucn)
            loss_ucn_pos_summary = tf.summary.scalar('ucn pos loss',
                                                     self.loss_ucn_pos)
            loss_ucn_neg_summary = tf.summary.scalar('ucn neg loss',
                                                     self.loss_ucn_neg)
            self.summary_op_src = tf.summary.merge([d_loss_src_summary,
                                                    g_loss_src_summary,
                                                    f_loss_src_summary,
                                                    loss_ucn_summary,
                                                    loss_ucn_pos_summary,
                                                    loss_ucn_neg_summary,
                                                    origin_images_summary,
                                                    sampled_images_summary])

            # target domain
            self.fx = self.content_extractor(self.trg_images, reuse=True)
            self.reconst_images = self.generator(self.fx, reuse=True)
            self.logits_fake = self.discriminator(self.reconst_images, reuse=True)
            self.logits_real = self.discriminator(self.trg_images, reuse=True)

            # loss
            self.d_loss_fake_trg = tf.losses.softmax_cross_entropy(
                class_two, self.logits_fake)      # L_D D2
            self.d_loss_real_trg = tf.losses.softmax_cross_entropy(
                class_three, self.logits_real)    # L_D D3
            self.d_loss_trg = self.d_loss_fake_trg + self.d_loss_real_trg
            self.g_loss_fake_trg = tf.losses.softmax_cross_entropy(
                class_three, self.logits_fake)  # L_GANG D3
            self.g_loss_const_trg = tf.reduce_mean(
                tf.square(self.trg_images - self.reconst_images)) * self.reconst_weight  # L_TID
            self.g_loss_trg = self.g_loss_fake_trg + self.g_loss_const_trg

            # optimizer
            self.d_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)

            # train op
            with tf.variable_scope('target_train_op', reuse=False):
                self.d_train_op_trg = slim.learning.create_train_op(
                    self.d_loss_trg, self.d_optimizer_trg, variables_to_train=d_vars)
                self.g_train_op_trg = slim.learning.create_train_op(
                    self.g_loss_trg, self.g_optimizer_trg, variables_to_train=g_vars)

            # summary op
            d_loss_fake_trg_summary = tf.summary.scalar(
                'trg_d_loss_fake', self.d_loss_fake_trg)
            d_loss_real_trg_summary = tf.summary.scalar(
                'trg_d_loss_real', self.d_loss_real_trg)
            d_loss_trg_summary = tf.summary.scalar('trg_d_loss',
                                                   self.d_loss_trg)
            g_loss_fake_trg_summary = tf.summary.scalar(
                'trg_g_loss_fake', self.g_loss_fake_trg)
            g_loss_const_trg_summary = tf.summary.scalar(
                'trg_g_loss_const', self.g_loss_const_trg)
            g_loss_trg_summary = tf.summary.scalar('trg_g_loss',
                                                   self.g_loss_trg)
            origin_images_summary = tf.summary.image('trg_origin_images',
                                                     self.trg_images)
            sampled_images_summary = tf.summary.image(
                'trg_reconstructed_images', self.reconst_images)
            self.summary_op_trg = tf.summary.merge([d_loss_trg_summary,
                                                    g_loss_trg_summary,
                                                    d_loss_fake_trg_summary,
                                                    d_loss_real_trg_summary,
                                                    g_loss_fake_trg_summary,
                                                    g_loss_const_trg_summary,
                                                    origin_images_summary,
                                                    sampled_images_summary])
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
