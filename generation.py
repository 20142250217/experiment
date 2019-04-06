# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:02:35 2019

@author: Administrator
"""

from train_process import *

if __name__ == '__main__':
    #读入源文件和目标文件
    #这是为了获取里面所有的单词，继而生成字典
    source_text,  vocab2int, int2vocab = loadData('train_subtile.txt', 'source')
    target_text,  _, _ = loadData('test_subtile.txt', 'target')


    #测试单词
    input_word = '你认为有可能是植物？'
    #得到最大长度
    max_length = len(input_word)
    #进行填充，实际上如果单词长度就是最大长度的话，不填充也行
    text = [vocab2int.get(char, vocab2int['<UNK>']) for char in jieba.lcut(input_word,cut_all=False)] + [vocab2int['<PAD>']] * (max_length - len(input_word))
    #打印输入的文字
    print(input_word)

    loaded_graph  = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        #读取模型
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        # 载入输入
        input_data = loaded_graph.get_tensor_by_name('encoder-input:0')
        encoder_sequence_length = loaded_graph.get_tensor_by_name('encoder_sequence_length:0')
        decoder_sequence_length = loaded_graph.get_tensor_by_name('decoder_sequence_length:0')
        predict = loaded_graph.get_tensor_by_name('predictions:0')


        feed = {
            input_data:[text] * batch_size,
            encoder_sequence_length:[len(input_word)] * batch_size,
            decoder_sequence_length:[len(input_word)] * batch_size,
        }

        #喂数据，预测
        int_curPred = sess.run([predict], feed_dict=feed)[0][0]
        #将输出的单词缩影转换为文本形式
        str_curPred = [int2vocab[int(num)] for num in int_curPred]

        #打印结果
        print('-----------------------')
        print('input:', text)
        print('output:', int_curPred)
        print('-----------------------')
        print('the input is:', input_word)
        print('the output is:', str_curPred)