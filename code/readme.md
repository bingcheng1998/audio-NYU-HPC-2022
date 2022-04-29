# 训练汉语拼音的wav2vec模型

## 数据集
- Primewords Chinese Corpus Set 1 https://www.openslr.org/resources/47/
- Free ST Chinese Mandarin Corpus ST-CMDS https://us.openslr.org/resources/38/
- SLR33 AISHELL-1 http://www.openslr.org/33/
- Aidatatang_200zh https://us.openslr.org/resources/62/

## TODO

- [ ] quartz 数据集制作，简化为同一个母类，使用transformer对数据预处理
- [ ] quartz + fft embedding dataset train 26 字母
- [ ] 拼音按照音素划分，符合163数据集的标注
- [ ] 重写dataloader，数据集+mask
- [ ] 训练quartz


- [ ] RNN用户特征提取
- [ ] quartz解码器 （论文里面这部分看起来很不对劲）




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



----
## 语音CTC注意事项
1. 只要loss不增加，初始的学习率尽量高（比如0.01），这样可以跳出local minimal
2. 维持高lr一段时间，直至收敛平稳后再坚持至少一个epoch，让梯度误差均匀分散
3. batch size在早期设置小一点，比如37，
   1. 如果太大，比如512，再配上0.01的lr，就不收敛了，预测的概率呈现均匀的横条状
   2. 如果太小，比如8，再配上0.01的lr就会老是反复，也不收敛，可以调低学习率达成收敛，但是慢
   3. 结论是恰好的bs配上恰好的lr才能有最好的结果
4. 初始阶段数据集首尾加上空白标签，辅助CTC loss学习，几个epoch后可能返回负值，去掉首尾空白标签即可正值