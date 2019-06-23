# -*- ecoding:utf-8 -*-
# @author:zhanglei
# @Time : 2019-04-05
# 处理数据的一些工具函数
#
import numpy as np
import pandas as pd


# check missing data
# fork from kaggle :https://www.kaggle.com/mjbahmani/top-3-nlp-libraries-tutorial-nltk-spacy-gensim
def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        # written by MJ Bahmani
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)