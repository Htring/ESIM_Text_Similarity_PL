# ESIM_Text_Similarity_PL
基于pytorch lightning的ESIM算法实现。具体实现可以参考我的博客:[【NLP】文本匹配——Enhanced LSTM for Natural Language Inference算法实现](https://blog.csdn.net/meiqi0538/article/details/124334676)
## 语料
实验数据选取，由于大部分数据是英文数据，但我更希望多做一些关于中文的内容。在github上一个开源项目:https://github.com/zhaogaofeng611/TextMatch.其数据集采用的是LCQMC数据，实现的模型在测试集上的效果：**ACC为0.8385**。

## ESIM实现

ESIM模型训练包含以下模块：

- 数据处理加载模块
- 模型实现模型
- pytorch_lightning 封装训练模块
- 模型训练和使用模块

## 模型训练与使用

模型训练了15个epoch，不适用与训练的字符向量的结果如下：

````text
Testing: 100%|██████████| 42/42 [00:03<00:00, 13.85it/s]
              precision    recall  f1-score   support

           0       0.78      0.94      0.85      6250
           1       0.92      0.73      0.81      6250

    accuracy                           0.83     12500
   macro avg       0.85      0.83      0.83     12500
weighted avg       0.85      0.83      0.83     12500

--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'accuracy': 0.8332800269126892,
 'f1_score': 0.8332800269126892,
 'recall': 0.8332800269126892,
 'val_loss': 0.4262129068374634}
--------------------------------------------------------------------------------
Testing: 100%|██████████| 42/42 [00:03<00:00, 13.19it/s]

Process finished with exit code 0

````

## 联系我

1. 我的github：[https://github.com/Htring](https://github.com/Htring)
2. 我的csdn：[科皮子菊](https://piqiandong.blog.csdn.net/)
3. 我订阅号：AIAS编程有道
   ![AIAS编程有道](https://mmbiz.qpic.cn/mmbiz_png/dQiaQ6INiazLqmEdj1NpUuAAUynfXekNte0cIG4lPcf38B0u4l1MYxNhGbQWdwKh4oPM0MI71hwkurerypzgPkyA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)
4. 知乎：[皮乾东](https://www.zhihu.com/people/piqiandong)





