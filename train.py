# -*- coding: utf-8 -*-
# @Time    : 4/25/18 8:14 AM

import tensorflow as tf
from read_utils import TextCoverter, batch_generator
import os
import codecs
from models import CharRNN

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'poetry_copy', 'name of the model')
tf.flags.DEFINE_string('input_file', './data/poetry_copy.txt', 'utf8 encoded text file')
tf.flags.DEFINE_integer('num_seqs', 2, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 4, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_integer('max_steps', 1000, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')

tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')


def main(_):
    # 设置模型的保存路径
    model_path = os.path.join('model', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    # 载入待训练文件
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()

    # 构建文字转换的实例
    converter = TextCoverter(text, FLAGS.max_vocab)
    # 保存已转换的文字实例的序列化数据，供后面的模型使用
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    # 将词转换成对应的词典中的位置的索引， 如“寒随穷律变，春逐鸟声开。初风飘带柳，晚雪间花梅。”, 因为','和句号在词典中排在前两位，
    # 则它们对应的索引是'0'和'1'，此处对应的arr即为[15 17 12 22 6 0 5 8 18 19 16 1 4 7 2 21 3 9 0 10 11 20 13 14 1]
    arr = converter.text_to_arr(text)
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)
    for x, y in g:
        print(x)
        print(y)
        break
    print("This is vocabulary size length: {}".format(converter.vocab_size))

    # 模型搭建
    model = CharRNN(converter.vocab_size,
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)

    # model.train(g,
    #             FLAGS.max_steps,
    #             model_path,
    #             FLAGS.save_every_n,
    #             FLAGS.log_every_n)


if __name__ == '__main__':
    tf.app.run()