# experiment
Model for our paper

seq2seq模型，使用了subtile数据。train_subtile,和test_subtile分别是用于输入和输出的数据。其句子是一一对应的复述。没有用中文词向量，embedding是随机初始化的。其中train_process是模型训练过程，generation是给定输入句子输出其复述的过程。从结果上来看十分粗糙。。。

接下来考虑使用预训练的中文词向量，然后在encoder端再加一个输入，用于接收另一个领域的句子，先加一个正交约束，看看效果有没有改进。
