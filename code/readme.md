# 训练汉语拼音的wav2vec模型

## 数据集
- Primewords Chinese Corpus Set 1 https://www.openslr.org/resources/47/
- Free ST Chinese Mandarin Corpus ST-CMDS https://us.openslr.org/resources/38/
- SLR33 AISHELL-1 http://www.openslr.org/33/

## TODO
- [x] 基础模型
- [x] 实验标准汉语拼音的训练
- [x] 多数据集联合
- [ ] 深入学习spleeter技术，掌握音频分离原理
- [ ] 制作音乐数据集
- [ ] 语音输入句子，自动识别单字，自动分割汉字，保存
- [ ] 修复最早用的那个带中断的模型，句子两端｜分割
- [ ] 多层汉字直接预测
- [ ] 在特征提取器后增加掩膜
- [ ] 在提取器后直接读取是否没有人声音，是的话可以进行句子划分