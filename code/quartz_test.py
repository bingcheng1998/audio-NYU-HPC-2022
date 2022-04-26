from model.quartznet import QuartzNet
from model.config import quartznet5x5_config

model = QuartzNet(quartznet5x5_config, feat_in = 80, vocab_size=36)

print(model)


# from utils.dataset import SpeechOceanDataset

# dataset = SpeechOceanDataset('./data/zhspeechocean/')
# train_set, test_set = dataset.split()

# print(train_set[0])