"""
http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
"""
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from random import shuffle

train_input = []
train_output = []
test_input = []
test_output = []

with open('./Data/train.csv') as f:
    train_temp = pd.read_table(f, sep=',')
    train_input_temp = train_temp.drop('subject',axis=1).as_matrix()
    for i in train_input_temp:
        train_input.append(i.reshape((9,1)))
    train_output_temp = train_temp['subject'].as_matrix()
    for i in train_output_temp:
        temp_list = ([0]*22)
        temp_list[int(i)-1] = 1
        train_output.append(temp_list)

with open('./Data/test.csv') as f:
    test_temp = pd.read_table(f, sep=',')
    test_input_temp = test_temp.drop('subject', axis=1).as_matrix()
    for i in test_input_temp:
        test_input.append(i.reshape((9,1)))
    test_output_temp = test_temp['subject'].as_matrix()
    for i in test_output_temp:
        temp_list = ([0]*22)
        temp_list[int(i)-1] = 1
        test_output.append(temp_list)

# Should have data in objects. Now to format (outputs to 'one hot' arrays)

data = tf.placeholder(tf.float32, [None, 9,1])
target = tf.placeholder(tf.float32, [None, 22])

def rnn(batch_size, epoch, num_hidden):
    # num_hidden = 24
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

    val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

    val = tf.transpose(val, [1,0,2])
    last = tf.gather(val,int(val.get_shape()[0]) - 1)

    weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

    cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)

    mistakes = tf.not_equal(tf.argmax(target,1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)

    # batch_size = 200

    no_of_batches = int(len(train_input)/batch_size)
    # epoch = 5

#    print('==========', 'Running', '==========')

    for i in range(epoch):
        ptr = 0
        for j in range(no_of_batches):
            inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
            ptr += batch_size
            sess.run(minimize, {data: inp, target:out})
        #print('Epoch - ', str(i))
    incorrect = sess.run(error, {data:test_input, target:test_output})

#    print('batch_size: ' , str(batch_size))
#    print('epoch: ', str(epoch))

    #print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
#    print('num_hidden: ', str(num_hidden))
#    print('accuracy: {:3.1f}%'.format((100 * (1- incorrect))))
    print(batch_size, epoch, num_hidden, '{:3.1f}'.format((100 * (1 - incorrect))), sep=',')
# print('=============================')

    sess.close()

if __name__ == '__main__':
    epoch = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    num_hidden = int(sys.argv[3])
    rnn(batch_size, epoch, num_hidden)
