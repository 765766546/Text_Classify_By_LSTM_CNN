用LSTM+CNN实现文本分类
数据集：
网上搜集的新闻数据，有10个类别categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']，每个分类6500条数据；

train.txt: 训练集(5000*10)

val.txt: 验证集(500*10)

test.txt: 测试集(1000*10)

训练所用的数据，以及训练好的词向量可以下载：链接: https://pan.baidu.com/s/1daGvDO4UBE5NVrcLaCGeqA 提取码: 9x3i

训练词向量

1.利用Lstm+Cnn进行文本分类
将LSTM与CNN连接在一起的关键：LSTM返回的值为[batch_size,seq_length,hidden_dim],而cnn需要的四维张量，故需要用到tf.expang_dims。

模型参数
parameters.py

预处理
预训练词向量进行embedding

对句子分词，去标点符号

去停用词

文字转数字

padding等

因为Cnn处理的是等长的序列，故在padding时，将所有句子padding到同一长度，本文指定最长序列max_length=300。

程序在data_processing.py

运行步骤
Training.py

由于运行非常吃力，因此只进行了3次迭代。但从迭代的效果来看，结果很理想。在训练集的batch中最好达到100%，同时测试集达到100%准确。

验证结果表明，5000条文本准确率达97.7%，取前10条语句的测试结果与原标签对比。