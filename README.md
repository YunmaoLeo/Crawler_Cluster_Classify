# PROJECT OVERVIEW

|INDEX|PROCESS|IMPLEMENT|
|----|-----|-------|
|1|创建基础爬虫框架|*send_request() parse_page()*<br>1) 使用*requests*获取连接，*BeautifulSoup*提取HTML信息 <br>2) 使用文本切片分别提取标题，求职贴正文|
|2|搭建多线程爬虫|*ShuimuSprider() Fast_Scrapy()*<br>1) 使用*thread,queue*创建多线程模块并结合爬虫工作函数 <br>2) 导出运行结果为Dataframe|*threading queue*| 
|3|数据清洗与整理|1) 去除岗位描述信息过短/无的数据 <br>2) *DataFrame.duplicated()*去重(存在重复广告) <br>3) 正则表达式提取岗位职责填写进入*requirement*列|
|4|文本分词及向量化|1) *preprocess()* 结合停用词字典清洗文本 <br>2) 使用jieba_fast进行分词 <br>3) *TfidVectorizer() TfidTransformer*文本向量化|
|5|降维|```KernelPCA(n_components=2,kernel='rbf')```
|6|聚类模型及簇数选择|1) 使用*MiniBatchKMeans*模型作为 <br>2) 使用*silhouette_score*协助进行簇数选择 <br>3) 聚类完成后分为4大类，0为金融相关岗位,1&3为计算机相关岗位，2为其他<br>(水木社区实习板块中金融与计算机领域的实习信息数量更多)|
|7|在给定聚类标签的情况下，搭建能分辨出金融、计算机岗位的分类器模型|*estimators=<br>('Tfid',TfidfVectorizer()),('KPCA',KernelPCA(n_components=100,kernel='rbf')),('xgb',xgboost.XGBClassifier(use_label_encoder=False))]<br><br>```pipeline = Pipeline(estimators)```<br><br>parameters=<br>{'KPCA__n_components':(4,50,100,150),<br>'xgb__max_depth':(8,7,6,5),<br>'xgb__n_estimators':(100,300,500)}<br><br>```grid_search = GridSearchCV(pipeline,parameters,n_jobs=-1,verbose=1)```
|8|模型评估|1) *Confusion Matrix* <br>2) *Classification_report*
|9|保存模型|```joblib.dump(model_clf,'Classification_model.joblib')```
  
<br>

# HOW TO USE IT

>[岗位爬虫及分类: `Scray_Classify.ipynb`](../codes/Scrapy_Classify.ipynb)
>+ 在Notebook内填写 `start` 与 `end` 参数，分别对应水木社区网页链接<br>"https://www.newsmth.net/nForum/#!article/Intern/`{id}`"<br>`id`字段的开始与结束
>+ code中默认去除岗位信息中无简历投递邮箱的数据，可根据自己需求进行调整
>+ 运行结束后将分别导出 `csv`, `txt` 结果


<br>

# Classification_report

>||precision|recall|f1-socre|support|
>|--|---------|------|--------|-------|
>|Finance|0.83|0.79|0.81|198|
>|IT|0.94|0.92|0.93|353|
>|Others|0.94|0.95|0.94|1065|
>| | | | | |
>|accuracy|nan|nan|0.92|1616|
>|macro avg|0.90|0.89|0.89|1616|
>|weighted avg|0.92|0.92|0.92|1616
<br><br>


## XBGOOST
>+ Introduction
>>`XGBoost` stands for `Extreme Gradient Boosting`.<br><br>
>>Instead of tweaking the instance weights at every iteration like AdaBoost,this method tries to fit new predirector to the `residual errors` made by the previous predictor.

>+ Hypterparameter
>> `learning_rate` scales the contribution of each tree. If it is a low value, such as 0.1, it will need more trees in ensemble to fit the set, but the predictions will usually generalize better.<br>
>> This is a regularization technique called `shrinkage收敛`

>> `warm_start=True` makes Sklearn keep existing trees when the `fit()` method is called, allowing incremental training.


## Stacking & Bagging
