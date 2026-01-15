Search Engine for Remote Homologous Proteins


Identifying proteins with similar structures in remote sequences is a difficult
undertaking. To address this issue, scientists have created a range of
techniques for executing remote homology searches. The aim of this project is
to create a deep learning-based algorithm that can identify up to 12 proteins
with similar structures from the Protein Data Bank (PDB) that are homologous to
a given protein sequence query.
在远程序列中识别具有相似结构的蛋白质是一项艰巨的任务。为了解决这个问题，科学家们创造了一系列执行远程同源性搜索的技术。
该项目的目的是创建一种基于深度学习的算法，该算法可以从蛋白质数据库（PDB）中识别多达12种结构相似的蛋白质，这些蛋白质与给定的蛋白质序列查询同源。

The quality of the pairing between the query and the candidate proteins is
assessed by computing the TM-score and SEQID between the query structure and
the paired PDB structure using the TMalign program (normalized by the query
sequence length). The final score is determined as follows:
通过使用 TMalign 程序计算查询结构和配对 PDB 结构之间的 TM 分数和 SEQID（通过查询序列长度标准化）来评估查询和候选蛋白质之间的配对质量。 最终得分确定如下：

        (TM-score - 0.6) + min(0.4 - SEQID, 0).

The effectiveness of the algorithm is evaluated by summing the total score of
all the query-candidate pairs.
通过对所有查询候选对的总分求和来评估算法的有效性。


#### query.fasta ####

This input file contains 1024 protein sequences that are to be used as queries.
Your program should take the file (in the same format but with different data)
as input and return up to 12 proteins that are similar to each query sequence.
该输入文件包含1024个将用作查询的蛋白质序列。您的程序应该将文件（格式相同但数据不同）作为输入，并返回最多12种与每个查询序列相似的蛋白质。

#### /data/pdb ####

This folder holds the Protein Data Bank of protein structures that can be
searched. The PDB files included can be accessed by a variety of Python
packages, including Graphein and BioPython.
此文件夹包含可搜索的蛋白质结构的蛋白质数据库。所包含的PDB文件可以由各种Python包访问，包括Graphein和BioPython。

#### tmalign.out ####

This training file has some potentially successful protein pairs with a
TM-score greater than 0.6 and a SEqID lower than 0.4.
该训练文件具有一些潜在成功的蛋白质对，其TM得分大于0.6，SEqID低于0.4。

#### result.out ####

This example output file has up to 12 potential proteins for each query
protein. Your program should generate output with the same format.
该示例输出文件对于每个查询蛋白质具有多达12个潜在蛋白质。您的程序应该生成具有相同格式的输出。

#### submission requirements ####

The final program should be submitted as a Docker (intructions to be added).

