https://scikit-learn.org/stable/index.html

https://github.com/amueller/scipy-2017-sklearn 指导手册

https://github.com/amueller/scipy-2017-sklearn 项目

https://pytorch.org/docs/stable/index.html pytorch 文档

https://pytorch.org/docs/stable/tensors.html pytorch 张量

https://blog.christianperone.com/2018/03/pytorch-internal-architecture-tour/ pytorch 内部结构

https://pytorch.org/docs/stable/notes/autograd.html 自动微分

https://medium.com/towards-data-science/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b pytorch 与 tensorflow 对比

https://github.com/pytorch/vision/tree/main/torchvision pytorch 视觉库

https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py pytorch 视觉库 alexnet 模型

https://github.com/tensorflow/serving tensorflow 服务器部署

https://ai.google.dev/edge/litert?hl=zh-cn tensorflow 轻量级部署 移动端

https://www.tensorflow.org/js?hl=zh-cn tensorflow js 部署
TensorFlow 的 JavaScript 版本，支持 GPU 硬件加速，可以运行在 Node.js 或浏览器环境中。它不但支持完全基于 JavaScript 从头开发、训练和部署模型，也可以用来运行已有的 Python 版 TensorFlow 模型，或者基于现有的模型进行继续训练。

https://pytorch.org/docs/stable/nn.html pytorch 神经网络库

https://pytorch.org/docs/nn.html#loss-functions pytorch 神经网络库 损失函数

视觉：
https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119 pytorch 神经网络库 提取特征

https://cs231n.github.io/neural-networks-3/ 神经网络

https://nbviewer.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb caffe 分类

https://blog.csdn.net/qq_27825451/article/details/90551513 使用 torch.nn.Sequential()一种整合各层的方法，同时可用，from collections import OrderDict, 一种带有时序信息的字典 使用 del 去删除
https://wenku.csdn.net/column/7asc6vxrxh
OrderDict

https://blog.csdn.net/coolend/article/details/102618637
numpy 创建一个视图 view(),reshape(x,y)，无法隔离 与 copy 新建一个完全的 base 互相隔离

https://blog.csdn.net/weixin_44544687/article/details/124703861
深度学习中的 Tensor 数据格式（N,C,H,W） 很有意义

- https://blog.csdn.net/m0_52650517/article/details/120662062

- https://blog.csdn.net/weixin_42426841/article/details/129903800?fromshare=blogdetail&sharetype=blogdetail&sharerId=129903800&sharerefer=PC&sharesource=2301_78911814&sharefrom=from_link
  PyTorch | torchvision.transforms 常用方法详解

```python

import os
import csv
import pandas as pd
from typing import List, Tuple
from bs4 import BeautifulSoup

def parse_index_page(index_path: str) -> List[str]:
    #TODO
    with open(index_path) as html_file:
        index_html = BeautifulSoup(html_file,'html.parser')
    genre_list = index_html.select('.genre-list li a')

    category_pages = []
    for category in genre_list:
        category_pages.append(category['href'])

    return category_pages

def parse_category_page(category_page_path: str, html_dir: str) -> List[str]:
    #TODO
    with open(category_page_path) as category_page_file:
        parse_category_html = BeautifulSoup(category_page_file,'html.parser')
    list = parse_category_html.select('.movie-item h2 a')
    category_path = []
    for category in list:
        new_path = os.path.join(html_dir,category['href'])
        category_path.append(new_path)

    return category_path

def parse_movie_page(movie_page_path: str) -> Tuple[str, str, str, str]:
    #TODO
    result = ()
    with open(movie_page_path) as movie_page_path_file :
        parse_movie_page_html = BeautifulSoup(movie_page_path_file,'html.parser')
    title_find = parse_movie_page_html.find('title')
    title_text = title_find.text
    # result.append(title_text)

    summart_section = parse_movie_page_html.find('section',class_='summary')
    Summary_find = summart_section.find('p')
    summary_text = Summary_find.text
    # result.append(summary_text)

    label_section = parse_movie_page_html.find('section',class_='details')
    label_find = label_section.find('p')
    label_text = label_find.text
    # result.append(label_text)

    time_find = parse_movie_page_html.find('h1')
    time_text = time_find.text
    time_text = time_text[:-1].split('(')[-1]
    # result.append(time_text)
    result =(title_text,summary_text,label_text,time_text)
    return result
def save_to_csv(movie_data: List[Tuple[str, str, str, str]], output_csv: str) -> None:
    #TODO

    return
def main(html_dir: str, output_csv: str) -> None:
    index_path = os.path.join(html_dir, 'index.html')

    category_pages = parse_index_page(index_path)
    print(f'{category_pages[:3]=}')
    movie_data = []

    for i, category_filename in enumerate(category_pages):
        category_page_path = os.path.join(html_dir, category_filename)

        movie_pages = parse_category_page(category_page_path, html_dir)
        if i == 0:
            print(f"{category_page_path=}, {movie_pages[:3]=}")
        for j, movie_page_path in enumerate(movie_pages):
            movie_info = parse_movie_page(movie_page_path)
            if i == 0 and j ==0:
                print(movie_info)
            movie_data.append(movie_info)

    save_to_csv(movie_data, output_csv)


if __name__ == "__main__":
    # html_dir = 'html_pages'
    html_dir = '/home/project/02/html_pages'
    output_csv = 'imdb_extracted.csv'

    main(html_dir, output_csv)


8 文本链式推理
import pickle
from typing import List, Tuple
import jieba
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import onnxruntime as ort


class TextSimilarityRecommender:
    def __init__(self, model_path: str, tokenizer_path: str):
        #TODO

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        #TODO

    def find_most_similar(self, query: str, corpus: List[str]) -> Tuple[str, float]:
        #TODO

    def recommend(self, query: str, corpus: List[str]) -> str:
        most_similar_text, similarity = self.find_most_similar(query, corpus)
        print(f"{most_similar_text=}, {similarity=}")
        highlighted_text = highlight_similar_words(query, most_similar_text)
        return highlighted_text


def highlight_similar_words(text1: str, text2: str) -> str:
    #TODO


if __name__ == '__main__':
    recommender = TextSimilarityRecommender("embedding.onnx", "embedding.pkl")
    corpus = ["北京今日天气：阴", "蓝桥杯赛事安排"]
    query = "北京今日天气怎么样"
    recommendation = recommender.recommend(query, corpus)
    print(recommendation)

9 MLM转换与推理
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
import onnxruntime as ort


def convert(model_path: str, tokenizer_path: str, onnx_path: str) -> None:
    #TODO


def inference(text: str, onnx_path: str, tokenizer_path: str) -> List[str]:
    #TODO


if __name__ == '__main__':
    model_path = "/home/project/09/bert-base-chinese"
    tokenizer_path = "/home/project/09/bert-base-chinese"
    onnx_path = "/home/project/09/bert_base_chinese.onnx"
    convert(model_path, tokenizer_path, onnx_path)
    text = "巴黎是 [MASK] 国的首都。"
    preds = inference(text, onnx_path, tokenizer_path)
    print(preds)

7 分组推理
import onnxruntime as ort
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict


def group_images_by_size(images_dict: Dict[str, np.ndarray]) -> Dict[Tuple[int, int], List[Tuple[str, np.ndarray]]]:
    #TODO


def grouped_inference(images_dict: Dict[str, np.ndarray], filename: str) -> Dict[str, np.ndarray]:
    grouped_images = group_images_by_size(images_dict)
    print([(size, len(images), images[0][1].shape) for size, images in grouped_images.items()])
    #TODO


def main() -> None:
    filename = 'srcnn.onnx'
    images = {f'image{i}': np.random.random([128*(i%3+1), 128*(i%2+1), 3]).astype(np.float32) for i in range(16)}
    print([(file, image.shape) for file, image in images.items()])
    images = grouped_inference(images, filename)
    print([(file, image.shape) for file, image in images.items()])


if __name__ == '__main__':
    main()

6 SGD调参
import torch
from torch.optim import Adam, SGD
from typing import Tuple

def rosenbrock_function(x: torch.Tensor, y: torch.Tensor, a: float=1., b: float=100.) -> torch.Tensor:

    return (a - x)**2 + b * (y - x**2)**2

def find_rosenbrock_minimum(max_iter: int) -> Tuple[float, float, float, SGD]:

    x = torch.tensor([0.0], requires_grad=True)
    y = torch.tensor([0.0], requires_grad=True)

    #TODO


if __name__ == "__main__":
    x_min, y_min, f_min, sgd = find_rosenbrock_minimum(max_iter=1000)
    print(f"局部最小值点：x = {x_min}, y = {y_min}, 对应的函数值：f(x, y) = {f_min}, SGD 参数：{sgd.state_dict()}")

5 点积注意力
import torch
from torch import nn, Tensor
import numpy as np
from typing import Tuple

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor)->Tuple[Tensor, Tensor]:
        #TODO
        pass

if __name__ == '__main__':
    scaled_dot_product_attn = ScaledDotProductAttention()
    d_model = 768
    batch_size, tgt_len, src_len = 2, 10, 20
    Q, K, V = torch.rand((batch_size, tgt_len, d_model)), torch.rand((batch_size, src_len, d_model)), torch.rand((batch_size, src_len, d_model))

    output1, output2 = scaled_dot_product_attn(Q, K, V)
    print(output1.shape) # torch.Size([2, 10, 20])
    print(output2.shape) # torch.Size([2, 10, 768])

```
