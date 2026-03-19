第一套代码是data.py、model.py、train.py、tools.py对应的日志文件是DualEncoder.txt
第二套代码是data1.py、model1.py、train1.py、tools.py对应的日志文件是DualEncoder_proj_gnn3.txt

现在存在的主要问题是：检索返回的蛋白质的Top1 score和Topk score都是负值。我的目标是最起码要全部是正的。根据公式score = tm_score - 0.6 + min(0.4 - seqid, 0.0)，score的取值范围是[-1.2,0.4],两端分别是空间完全不相似、序列100%相似和空间完全相似、序列相似度低于40%。

我怀疑batch的构造形式和训练的目标出了问题。即如何构建pair，如何制定损失函数。在我的训练数据中，所有的(query,candidate)对的score都是大于等于0的，且从0到0.4程线性下降分布。所以我不认为in-batch negatives是合适的。

你应该提出合适的batch构建方法以及loss构建方法，我觉得模型不需要太复杂，同样的，batch和loss的构建方法也不需要太复杂。现在的检索结果为负值应该是有一个很重要的东西没做对。

请你给出方案。