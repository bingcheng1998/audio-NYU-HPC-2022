# 训练汉语拼音的wav2vec模型

## 数据集
- Primewords Chinese Corpus Set 1 https://www.openslr.org/resources/47/ 有用户区分
- Free ST Chinese Mandarin Corpus ST-CMDS https://us.openslr.org/resources/38/
- SLR33 AISHELL-1 http://www.openslr.org/33/
- Aidatatang_200zh https://us.openslr.org/resources/62/
- 说话人识别 http://www.openslr.org/82/
- 多说话人语音合成 http://www.openslr.org/93/

## TODO

- [x] quartz 数据集制作，简化为同一个母类，使用transformer对数据预处理
- [x] quartz + fft embedding dataset train 26 字母
- [x] 拼音按照音素划分，符合163数据集的标注
- [x] 重写dataloader，数据集+mask
- [x] 训练quartz
- [x] RNN用户特征提取
- [x] quartz解码器
- [x] 基础模型
- [x] 实验标准汉语拼音的训练
- [x] 多数据集联合
- [x] 音频去掉静音区
- [x] 在传统tacotron2上小batch训练单人中文数据
- [ ] 训练speaker encoder，进行多说话人训练
- [x] datatang 和 dataaishell3的loader
- [ ] tacotron2 训练+添加用户音色模块
- [x] 深入学习spleeter技术，掌握音频分离原理(U-Net, fft, ifft 没有使用深度学习decoder)
- [ ] 制作音乐数据集, 爬取同一首歌的多个演唱版本，然后制作数据集
- [ ] 语音输入句子，自动识别单字，自动分割汉字，保存
- [ ] 修复最早用的那个带中断的模型，句子两端｜分割
- [ ] 多层汉字直接预测
- [x] 在特征提取器后增加掩膜
- [ ] 在提取器后直接读取是否没有人声音，是的话可以进行句子划分
- [ ] 日韩语言联合训练（同一个aux？不同的aux？）
- [x] 根据论文https://arxiv.org/pdf/1904.10619.pdf，然而我们无法修改ctcloss（因为是c++写的，修改需要重新编译），使用迂回方法，减少空白标签占比，增加一个loss，直接降低空白标签。叫做blank_loss, 使用margin，削掉最大的1/5blank的分值，然后最小化其余4/5的blank的分值。
- [ ] 单字分离算法，不使用动态规划，写个robust的naive算法，brute force也可以
- [x] 中文两边加‘｜’，然后最小化空格‘-’，阈值最小化‘｜’
- [ ] 静音区推断，然后静音区使用‘｜’，防止x, s等气音被误识别为‘｜’
- [ ] 使用tacotron2，但是不去记忆用户embedding，而是使用前一段话做一个引子，单字生成，输入单字的拼音，音调，时常，然后自动生成单个字的发音
- [ ] 先实现多说话人声音的语音合成，和上面原理一致，因为输入的一句话会有语气，所以很可能可以顺便学到语气
- [ ] 实现困难太大，尝试在原模型上训练过程抵消掉原始mel的指导过强的影响，可以使用一个参数alpha来配置需要原始mel和现实预测mel的比率



----
## 语音CTC注意事项
1. 只要loss不增加，初始的学习率尽量高（比如0.01），这样可以跳出local minimal
2. 维持高lr一段时间，直至收敛平稳后再坚持至少一个epoch，让梯度误差均匀分散
3. batch size在早期设置小一点，比如37，
   1. 如果太大，比如512，再配上0.01的lr，就不收敛了，预测的概率呈现均匀的横条状
   2. 如果太小，比如8，再配上0.01的lr就会老是反复，也不收敛，可以调低学习率达成收敛，但是慢
   3. 结论是恰好的bs配上恰好的lr才能有最好的结果
4. 初始阶段数据集首尾加上空白标签，辅助CTC loss学习，几个epoch后可能返回负值，去掉首尾空白标签即可正值


mul-ST-CMDS.pt
data_aishell.pt
mul-aidatatang.pt