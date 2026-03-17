请你根据utils下的py文件、以及当前文件夹下的train.py文件、-4lr_clip_selfmask_topm_no_selfhit.txt为我解决以下问题：
1.分析我的模型loss很小，但是由score = tm_score - 0.6 + min(0.4 - seqid, 0.0)计算出来的远程同源性分数为负值的原因。我的猜测是模型只学到了序列相似性，所以返回了seqid>0.4的序列，而TM-score语言模型学不到结构信息，所以返回的结果中TM-score<0.6
2.在每一轮测试中，(Average Top1 TMscore-0.6) + min(0.4-Average Top-1 SeqID,0) 不等于Average Top-1 Remote homologous score。请你分析是哪些函数出现了问题.
3.分析完成之后，你应该提出相应的解决方案，并给予代码实现，请你不要直接在原文件上修改我的代码。