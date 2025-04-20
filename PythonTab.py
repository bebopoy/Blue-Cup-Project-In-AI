# # 内容损失函数
# def content_loss(Y_hat, Y):
#     return torch.pow(Y_hat - Y.detach(), 2).mean()

# # 风格损失函数
# def style_loss(Y_hat, gram_Y):
#     # return torch.square(gram(Y_hat) - gram_Y.detach()).mean()     # 新版本PyTorch使用此行
#     return torch.pow(gram(Y_hat) - gram_Y.detach(), 2).mean()

# for epoch in range(num_epochs):
#     #TODO
#     trainer.zero_grad()
#     contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers, net)
#     contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
#     l.backward()
#     trainer.step()
#     scheduler.step()

#     if (epoch + 1) % 25 == 0:
#         plt.imshow(postprocess(X))
#         plt.show()
#     print(f'Epoch {epoch + 1}, Content Loss: {float(sum(contents_l))}, Style Loss: {float(sum(styles_l))}, TV Loss: {float(tv_l)}')

#     # 在最后一轮训练后，将损失函数的结果保存到文件中
#     if epoch == num_epochs - 1:
#         with open('output_d.txt', 'w') as f:
#             f.write(f'{float(sum(contents_l))}, {float(sum(styles_l))}, {float(tv_l)}\n')
# output_img = postprocess(X)
# output_img.save('output_d.jpg')

import torch
import torchvision
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt


# 初始化合成图像，定义生成图像的类
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


# 图像预处理函数
def preprocess(img, image_shape):
    rgb_mean = torch.tensor([0.485, 0.456, 0.406])
    rgb_std = torch.tensor([0.229, 0.224, 0.225])
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)


# 图像后处理函数
def postprocess(img):
    # 定义图像预处理和后处理的参数
    rgb_mean = torch.tensor([0.485, 0.456, 0.406])
    rgb_std = torch.tensor([0.229, 0.224, 0.225])
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))


# 特征提取函数
def extract_features(X, content_layers, style_layers, net):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


# 获取内容图像特征的函数
def get_contents(content_img, image_shape, device, content_layers, style_layers, net):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers, net)
    return content_X, contents_Y


# 获取风格图像特征的函数
def get_styles(style_img, image_shape, device, content_layers, style_layers, net):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers, net)
    return style_X, styles_Y


# 内容损失函数
def content_loss(Y_hat, Y):
    return torch.pow(Y_hat - Y.detach(), 2).mean()


# Gram矩阵函数，用于计算风格损失
def gram(X):
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)


# 风格损失函数
def style_loss(Y_hat, gram_Y):
    # return torch.square(gram(Y_hat) - gram_Y.detach()).mean()     # 新版本PyTorch使用此行
    return torch.pow(gram(Y_hat) - gram_Y.detach(), 2).mean()


# 全变分损失，用于平滑生成的图像
def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


# 总损失函数
def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分配权重
    content_weight, style_weight, tv_weight = 1, 1e3, 10

    # 分别计算内容损失、风格损失和全变分损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l


# 一些必要的初始化
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer


# 训练模型
def train(X, contents_Y, styles_Y, content_layers, style_layers, net, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)

    for epoch in range(num_epochs):
        #TODO
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers, net)
        contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()

        if (epoch + 1) % 25 == 0:
            plt.imshow(postprocess(X))
            plt.show()
        print(f'Epoch {epoch + 1}, Content Loss: {float(sum(contents_l))}, Style Loss: {float(sum(styles_l))}, TV Loss: {float(tv_l)}')

        # 在最后一轮训练后，将损失函数的结果保存到文件中
        if epoch == num_epochs - 1:
            with open('output_d.txt', 'w') as f:
                f.write(f'{float(sum(contents_l))}, {float(sum(styles_l))}, {float(tv_l)}\n')
    output_img = postprocess(X)
    output_img.save('output_d.jpg')


def main():
    # 加载内容图像和风格图像
    content_img = Image.open('img_d.jpg')
    style_img = Image.open('style_d.jpg')

    # 显示内容图像和风格图像
    plt.imshow(content_img)
    plt.show()
    plt.imshow(style_img)
    plt.show()

    # 加载预训练的VGG19模型
    model_path = "vgg19-dcbb9e9d.pth"
    pretrained_net = torchvision.models.vgg19(pretrained=False)
    pretrained_net.load_state_dict(torch.load(model_path))

    # 定义风格层和内容层
    style_layers, content_layers = [0, 5, 10, 19, 28], [25]

    # 创建一个新的网络，该网络只包含VGG19的前几层
    net = nn.Sequential(*[pretrained_net.features[i] for i in
                        range(max(content_layers + style_layers) + 1)])


    # 定义超参数并训练模型
    device, image_shape = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), (300, 450)
    net = net.to(device)
    content_X, contents_Y = get_contents(content_img, image_shape, device, content_layers, style_layers, net)
    _, styles_Y = get_styles(style_img, image_shape, device, content_layers, style_layers, net)
    train(content_X, contents_Y, styles_Y, content_layers, style_layers, net, device, 0.3, 50, 10)


if __name__ == '__main__':
    main()


# tensorfolw 模型定义
import numpy as np
import jieba
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def preprocess_text(data_file):
    data = []
    with open(data_file, "r") as f:
        for text in f.readlines():
            for sentence in text.strip().split("。"):
                data.append(jieba.lcut(sentence))
    return data

def build_vocab(data):
    vocab = set()
    for seq in data:
        for word in seq:
            vocab.add(word)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_size = len(vocab)
    return word_to_idx, idx_to_word, vocab_size

def preprocess_data(data, word_to_idx):
    input_data = []
    output_data = []
    for seq in data:
        input_seq = [word_to_idx[w] for w in seq[:len(seq)-1]]
        output_seq = [word_to_idx[w] for w in seq[1:len(seq)]]
        input_data.append(input_seq)
        output_data.append(output_seq)
    input_data_padded = pad_sequences(input_data, maxlen=50)
    output_data_padded = pad_sequences(output_data, maxlen=50)
    output_sequence_onehot = tf.one_hot(output_data_padded, depth=len(word_to_idx))

    return input_data_padded, output_sequence_onehot

def build_train_model(vocab_size, input_data, output_data, model_save_path):
    #TODO
    embedding_dim=128
    hidden_units=128

    model = Sequential()  # 创建一个序贯模型
    # 添加嵌入层，将输入的单词索引映射为embedding_dim维的向量表示
    model.add(Embedding(vocab_size, embedding_dim, input_length=None))
    # 添加一个LSTM层，将嵌入层的输出作为输入，输出序列的每个时间步都包含完整的隐藏状态序列
    model.add(LSTM(hidden_units, return_sequences=True))
    # 添加一个全连接层，输出维度为vocab_size，使用softmax激活函数进行分类
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')  # 编译模型，配置优化器和损失函数
    # 训练模型
    callback = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(input_data, output_data, batch_size=2, epochs=3, callbacks=[callback])
    model.save(model_save_path)  # 保存训练好的模型

def generate_text(model_save_path, start_sequence, word_to_idx, idx_to_word):
    #TODO
    generated_text = start_sequence  # 初始化生成的文本为起始序列
    model = load_model(model_save_path)  # 加载训练好的模型
    #input_seq = [word_to_idx[word] for word in generated_text.split()]  # 将生成的文本转换为单词索引序列
    input_seq = [word_to_idx[generated_text]]
    input_seq = np.array(input_seq).reshape(1, -1)  # 对输入序列进行预测
    predictions = model.predict(input_seq)
    # 获取预测结果中概率最高的索引
    pred_idx = np.argmax(predictions)
    word = idx_to_word[pred_idx]  # 根据预测的单词索引找到对应的单词
    generated_text += " " + word  # 将生成的单词添加到生成的文本中
    return generated_text

def main():
    data_file = "lm_train_data.txt"
    model_save_path = "llm_model.h5"
    start_sequence = "美"
    train_data = preprocess_text(data_file)
    # 由于全部数据内存太大，此处只取前 1000 条数据进行训练
    train_data = train_data[:1000]
    word_to_idx, idx_to_word, vocab_size = build_vocab(train_data)
    input_data, output_data = preprocess_data(train_data, word_to_idx)
    build_train_model(vocab_size, input_data, output_data, model_save_path)
    generated_text = generate_text(model_save_path, start_sequence, word_to_idx, idx_to_word)
    print(generated_text)

if __name__ == '__main__':
    main()