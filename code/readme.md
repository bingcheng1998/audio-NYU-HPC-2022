# 训练汉语拼音的wav2vec模型

## 数据集
- Primewords Chinese Corpus Set 1 https://www.openslr.org/resources/47/
- Free ST Chinese Mandarin Corpus ST-CMDS https://us.openslr.org/resources/38/
- SLR33 AISHELL-1 http://www.openslr.org/33/

## TODO
- [x] 基础模型
- [ ] 多数据集联合
- [ ] 在特征提取器后增加掩膜
- [ ] 实验标准汉语拼音的训练
- [ ] 在提取器后直接读取是否没有人声音，是的话可以进行句子划分