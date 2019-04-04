import tensorflow as tf
from data_processing import batch_iter, seq_length
from Parameters import Parameters as pm

class Lstm_CNN(object):

    def __init__(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, pm.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, pm.num_classes], name='input_y')
        self.length = tf.placeholder(tf.int32, shape=[None], name='rnn_length')
        self.keep_pro = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.lstm_cnn()

    def lstm_cnn(self):
		#embedding层
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding = tf.get_variable("embeddings", shape=[pm.vocab_size, pm.embedding_dim],
                                             initializer=tf.constant_initializer(pm.pre_trianing))
            embedding_input = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope('LSTM'):
            cell = tf.nn.rnn_cell.LSTMCell(pm.hidden_dim, state_is_tuple=True)
            Cell = tf.contrib.rnn.DropoutWrapper(cell, self.keep_pro)
            output, _ = tf.nn.dynamic_rnn(cell=Cell, inputs=embedding_input, sequence_length=self.length, dtype=tf.float32)
		
		'''
		卷积层filte 高度设定为【2，3，4】三种，宽度与词向量等宽，
		卷积核数量设为num_filter。假设batch_size =1，即对一个句子进行卷积操作。
		每一种filter卷积后，结果输出为[1,seq_length - filter_size +1,1,num_filter]的tensor。
		再用ksize=[1,seq_length - filter_size + 1,1,1]进行max_pooling,得到[1,1,1,num_filter]这样的tensor.
		将得到的三种结果进行组合,得到[1,1,1,num_filter*3]的tensor.最后将结果变形一下[-1,num_filter*3]，
		目的是为了下面的全连接。
		'''
        with tf.name_scope('CNN'):
            outputs = tf.expand_dims(output, -1) #[batch_size, seq_length, hidden_dim, 1]
            pooled_outputs = []
            for i, filter_size in enumerate(pm.filters_size):
                filter_shape = [filter_size, pm.hidden_dim, 1, pm.num_filters]
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='w')
                b = tf.Variable(tf.constant(0.1, shape=[pm.num_filters]), name='b')
                conv = tf.nn.conv2d(outputs, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                pooled = tf.nn.max_pool(h, ksize=[1, pm.seq_length-filter_size+1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name='pool')
                pooled_outputs.append(pooled)
            output_ = tf.concat(pooled_outputs, 3)
            self.output = tf.reshape(output_, shape=[-1, 3*pm.num_filters])
		'''
		全连接层
		在全连接层中进行dropout,通常保持率为0.5。其中num_classes为文本分类的类别数目。然后得到输出的结果scores，
		以及得到预测类别在标签词典中对应的数值predicitons。
		'''
        with tf.name_scope('output'):
            out_final = tf.nn.dropout(self.output, keep_prob=self.keep_pro)
            o_w = tf.Variable(tf.truncated_normal([3*pm.num_filters, pm.num_classes], stddev=0.1), name='o_w')
            o_b = tf.Variable(tf.constant(0.1, shape=[pm.num_classes]), name='o_b')
            self.logits = tf.matmul(out_final, o_w) + o_b
            self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='score')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
		
		'''
		optimizer
		这里使用了梯度裁剪。首先计算梯度，这个计算是类似L2正则化计算w的值，也就是求平方再平方根。
		然后与设定的clip裁剪值进行比较，如果小于等于clip,梯度不变；如果大于clip,则梯度*（clip/梯度L2值）
		'''
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(pm.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            # 对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            # global_step 自动+1

        with tf.name_scope('accuracy'):
            correct = tf.equal(self.predict, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    def feed_data(self,x_batch, y_batch, real_seq_len, keep_pro):
        feed_dict ={self.input_x: x_batch,
                    self.input_y: y_batch,
                    self.length: real_seq_len,
                    self.keep_pro: keep_pro}
        return feed_dict

    def test(self,sess, x, y):
        batch_test = batch_iter(x, y, batch_size=pm.batch_size)
        for x_batch, y_batch in batch_test:
            real_seq_len = seq_length(x_batch)
            feed_dict = self.feed_data(x_batch, y_batch, real_seq_len, 1.0)
            test_loss, test_accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)

        return test_loss, test_accuracy





