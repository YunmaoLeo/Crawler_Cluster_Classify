  |Contents|
|--------|
|[PROJECT OVERVIEW](#project-overview)|
|[HOW TO USE IT](#how-to-use-it)|
|[CLASSIFICATION MODEL](#classificationmodel)|
|[FILE MENU](#filemenu)
|[RELATED INFO](#related-info)

# PROJECT OVERVIEW

|INDEX|PROCESS|IMPLEMENT|
|----|-----|-------|
|1|创建基础爬虫框架|*send_request() parse_page()*<br>1) 使用*requests*获取连接，*BeautifulSoup*提取HTML信息 <br>2) 使用文本切片分别提取标题，求职贴正文|
|2|搭建多线程爬虫|*ShuimuSprider() Fast_Scrapy()*<br>1) 使用*thread,queue*创建多线程模块并结合爬虫工作函数 <br>2) 导出运行结果为Dataframe|*threading queue*| 
|3|数据清洗与整理|1) 去除岗位描述信息过短/无的数据 <br>2) *DataFrame.duplicated()*去重(存在重复广告) <br>3) 正则表达式提取岗位职责填写进入*requirement*列|
|4|文本分词及向量化|1) *preprocess()* 结合停用词字典清洗文本 <br>2) 使用jieba_fast进行分词 <br>3) *TfidVectorizer() TfidTransformer*文本向量化|
|(4.5)|尝试使用KernelPCA降维后使用MiniBatchKMeans做聚类，但发现聚类结果中不同样本数量偏差过大，但能显著出金融、计算机相关岗位的类别，可能是因为这两类的数量居多|``待解决``|
|5|根据岗位信息标题进行标注,共分10类|```df['TARGET']=(df['title_no_bracket']).apply(lambda x: label_mark(x))```|
|6|在给定标签的情况下，搭建Xgboost分类器模型|```estimators=<br>('Tfid',TfidfVectorizer()),('KPCA',KernelPCA(n_components=100,kernel='rbf')),('xgb',xgboost.XGBClassifier(use_label_encoder=False))]```<br><br>```pipeline = Pipeline(estimators)```|
|7|模型评估|0) *sklearn.model_selection.cross_val_predict*<br>1) *Confusion Matrix* <br>2) *Classification_report*
|8|保存模型|```joblib.dump(model_clf,'Classification_model.joblib')```

  
<br>

# HOW TO USE IT

>[岗位爬虫及分类: `Scray_Classify.ipynb`](../codes/Scrapy_Classify.ipynb)
>+ 在Notebook内填写 `start` 与 `end` 参数，分别对应水木社区网页链接<br>"https://www.newsmth.net/nForum/#!article/Intern/`{id}`"<br>`id`字段的开始与结束
>+ code中默认去除岗位信息中无简历投递邮箱的数据，可根据自己需求进行调整
>+ 运行结束后将分别导出 `csv`, `txt` 结果


<br>

# ClassificationModel
## LabelMark


>```
>def label_mark(x):
>    result =''
>    for key,val in LabelTransfer.items():
>        if(re.findall(val,x)!=[]):
>            result=key
>    return result 
> df['TARGET']=(df['title_no_bracket']).apply(lambda x: label_mark(x))
> ```
> 
>```{ 
>'其他':'(英语|外语|日语|韩语|德语|法语|葡萄牙语|公关|销售|媒体|设计|编辑|内容|培训|市场|营销|客服|行政|助理|视频|法律|法务财务|会计|审计|税务|出版社)',  
>'人力资源':'(HR|人事|人力|猎头)',  
>'运营实习':'(运营|产品|新媒体)',  
>'投资咨询实习生':'(投资|资本|并购|咨询)',  
>'算法实习':'(算法|搜索|自然语言|NLP|CV|图像处理|机器学习|深度学习|AI)',  
>'券商投行基金':'(券商|证券|投行|资管|金融|信托|资产|新财富|IBD|IPO|期权|基金|研究员)',  
>'前端&测试':'(前端|测试)',
>'量化交易':'(量化|Quant)',
>'研发开发':'(研发|开发|软件工程师|后端|云计算|Python|Java|java|C\+\+)',   
>'数据分析挖掘':'(数据分析|数据挖掘|商业分析)',  
>}```

## LabeledDataset
>|Column|Count|
>|------|-----|
>券商投行基金 |    3733
>算法实习   |    3229
>研发开发  |     3090
>其他       |  2557
>运营实习     |  2319
>投资咨询实习生 |   2308
>人力资源  |     1401
>前端&测试|      1399
>数据分析挖掘|      854
>量化交易|        577

``issue: Inbalanced distribution of samples``

## Xgboost Model with Pipeline
```
Pipeline(steps=[('Tfid', TfidfVectorizer()),
                ('KPCA', KernelPCA(kernel='rbf', n_components=100)),
                ('xgb',
                 XGBClassifier(base_score=0.5, booster='gbtree',
                               colsample_bylevel=1, colsample_bynode=1,
                               colsample_bytree=1, gamma=0, gpu_id=-1,
                               importance_type='gain',
                               interaction_constraints='',
                               learning_rate=0.300000012, max_delta_step=0,
                               max_depth=6, min_child_weight=1, missing=nan,
                               monotone_constraints='()', n_estimators=100,
                               n_jobs=12, num_parallel_tree=1,
                               objective='multi:softprob', random_state=0,
                               reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
                               subsample=1, tree_method='exact',
                               use_label_encoder=False, validate_parameters=1,
                               verbosity=None))])
```
``RUNTIME: 2min 56s``

## Classification_report

>
                    precision    recall  f1-score   support

        人力资源       0.82      0.79      0.80       261
          其他       0.66      0.65      0.66       492
      券商投行基金       0.79      0.83      0.81       716
       前端&测试       0.85      0.81      0.83       285
     投资咨询实习生       0.77      0.69      0.72       472
      数据分析挖掘       0.71      0.48      0.57       172
        研发开发       0.77      0.80      0.78       613
        算法实习       0.82      0.82      0.82       640
        运营实习       0.74      0.85      0.79       452
        量化交易       0.78      0.82      0.80        97

    accuracy                           0.77      4200
    macro avg       0.77      0.75      0.76      4200
    weighted avg       0.77      0.77      0.77      4200
>
<br><br>

[Model.joblib link](../models/Classification_model_5_n_estimator=100_pca=400.joblib)  


# FILEMENU
|Folder|Files|
|-----------|-----|
|``codes``|``Scrapy_Classify.ipynb``:爬虫+岗位分类<br>``Scrapy_Training.ipynb``数据整理+多分类模型训练|
|``datasets``|存放文件导出的csv,txt|
|``models``|分类模型|
|``stopwords``|``baidu_stopwords``:自己增加了一些有关招聘信息文本处理的停用词

# RELATED INFO

## XBGOOST
>+ Introduction
>>`XGBoost` stands for `Extreme Gradient Boosting`.<br><br>
>>Instead of tweaking the instance weights at every iteration like AdaBoost,this method tries to fit new predirector to the `residual errors` made by the previous predictor.

>+ Hypterparameter
>> `learning_rate` scales the contribution of each tree. If it is a low value, such as 0.1, it will need more trees in ensemble to fit the set, but the predictions will usually generalize better.<br>
>> This is a regularization technique called `shrinkage收敛`

>> `warm_start=True` makes Sklearn keep existing trees when the `fit()` method is called, allowing incremental training.


## Stacking & Bagging
