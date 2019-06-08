import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from pylab import mpl  
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
import itertools
from  collections import Counter

mpl.rcParams['font.sans-serif'] = ['SimHei'] 
#指定默认字体

#设置Path（图片存储路径）
cur_dir = os.getcwd() 
folder_name = '图'
dir_new=os.path.join(cur_dir, folder_name)
if( os.path.exists(dir_new) == False ):
    path=os.mkdir(dir_new)
path=dir_new+'\\'

shizhong = pd.read_csv('jinan_sz.csv')  # 读取wenjian
huaiyin = pd.read_csv('jinan_hy.csv') 
lixia = pd.read_csv('jinan_lx.csv') 
licheng = pd.read_csv('jinan_lc.csv') 
gaoxin = pd.read_csv('jinan_gx.csv') 
tianqiao = pd.read_csv('jinan_tq.csv') 
changqing = pd.read_csv('jinan_cq.csv') 
zhangqiu = pd.read_csv('jinan_zq.csv')

'''
方法名称：qingxi
功能：    清洗数据，方便分析和机器学习
参数：        df    数据
返回：        df    清洗后的数据
'''
def qingxi(df):
    df[["室","厅"]] = df["类型"].str.extract("(\d+)室(\d+)厅")
    df["室"] = df["室"].astype(float)
    df["厅"] = df["厅"].astype(float)
    df["面积"] = df["面积"].astype(float)
    df["总价"] = df["总价"].map(lambda e:e.replace("万",""))
    df["总价"] = df["总价"].astype(float)
    df["单价（元/m2）"] = df["单价（元/m2）"].astype(float)
    df["朝向"]=df["朝向"].map(lambda e:e.replace("向",""))
    df['朝向']=df["朝向"].map(lambda e:e.replace("南北","南"))#设置南北朝向为南朝向
    df['朝向']=df["朝向"].map(lambda e:e.replace("东西","东"))#设置东西朝向为东朝向
    df_direction = pd.get_dummies(df["朝向"])
    
    df["楼层"] = df["楼层"].map(lambda e:e.split('（')[0])
    df_floor = pd.get_dummies(df["楼层"])

    #将朝向，楼层的dummy variable 拼接到df
    df = pd.concat([df,df_direction,df_floor],axis=1)
    try:
        df['东北']
    except:
        df['东北']=0
    try:
        df['东南']
    except:
        df['东南']=0
    try:
        df['中层']
    except:
        df['中层']=0
    try:
        df['西北']
    except:
        df['西北']=0
    try:
        df['西南']
    except:
        df['西南']=0
    try:
        df['中层']
    except:
        df['中层']=0
    #删除“北”，“中层”，,删除逻辑错误的数据，无用及多重共线性的数据
    try:
        
        del df["类型"]
        # del df["小区名称"]
        del df["朝向"]
        del df["楼层"]
        del df["北"]
        del df["中层"]
    except:
        print('删除失败')
    return df
#清洗各个数据
shizhong=qingxi(shizhong)
huaiyin=qingxi(huaiyin)
lixia=qingxi(lixia)
licheng=qingxi(licheng)
gaoxin=qingxi(gaoxin)
tianqiao=qingxi(tianqiao)
changqing=qingxi(changqing)
zhangqiu=qingxi(zhangqiu)


#绘制各个行政区的二手房平均房价
qu=['市中','槐荫','历下','历城','高新','天桥','长清','章丘']
zongjia=[sum(shizhong['总价'])/len(shizhong),sum(huaiyin['总价'])/len(huaiyin),sum(lixia['总价'])/len(lixia),
          sum(licheng['总价'])/len(licheng),sum(gaoxin['总价'])/len(gaoxin),sum(tianqiao['总价'])/len(tianqiao),
          sum(changqing['总价'])/len(changqing),sum(zhangqiu['总价'])/len(zhangqiu)] #平均总房价

pingjunmianji=(sum(shizhong['面积'])+sum(huaiyin['面积'])+sum(lixia['面积'])+sum(licheng['面积'])+sum(gaoxin['面积'])+sum(tianqiao['面积'])+sum(changqing['面积'])+sum(zhangqiu['面积']))/(len(shizhong)+len(huaiyin)+len(lixia)+len(licheng)+len(gaoxin)+len(tianqiao)+len(changqing)+len(zhangqiu))
#平均面积
#画出各区平均房价
plt.cla()
plt.bar(qu,zongjia)
plt.title('各个行政区的二手房平均房价')
plt.savefig(path+'各个行政区的二手房平均房价.png')

'''
方法名称：  xiaoqu
功能：      画出各个小区的平均房价
参数：        df    数据
              st    区名
返回         maxxiaoqu,xiaoqu    总价最贵的小区（总价，小区名称）
'''

def xiaoqu(df,st):
    df=df[df['面积']<=300]#去除丽群孤立数据
    df=df[df['总价']<=1000]
    maxxiaoqu=0
    xiaoqu=''
    plt.rcParams['savefig.dpi'] = 300 #图片像素  默认figure
    plt.rcParams['figure.dpi'] = 300 #分辨率    默认72
    dq=[]
    fj=[]
    q=df['小区名称'].drop_duplicates(keep='first')#获取小区列表
    for i in q:
        m=0
        s=0
        f=df[df['小区名称']==i]
        for index,j in f.iterrows():
            if 10<int(j['总价']):
                m+=1
                s+=int(j['总价'])
            else:
                continue
            #print(j[0]+"平均租房价格为："+str(int(s/m)))
            #f[j[0]]=int(s/m)
        dq.append(i)
        fj.append(int(s/m))
        if maxxiaoqu<int(s/m):
            maxxiaoqu=int(s/m)
            xiaoqu=i
    #设置画板大小
    plt.figure(figsize=(22,7))
    plt.cla()#清空画板
    plt.title(st+'区\n各个小区的平均房价（不考虑面积）')
    plt.xticks(range(0,len(dq),1),rotation=-90,fontsize=2.5)#设置x轴刻度标签方向及大小
    plt.bar(dq,fj)
    plt.savefig(path+'各个小区的平均房价_'+st+'.png')
    plt.rcParams['savefig.dpi'] = 'figure' #图片像素  默认figure
    plt.rcParams['figure.dpi'] = 72 #分辨率    默认72
    return maxxiaoqu,xiaoqu   
#求出各个小区的平均房价

maxxiaoqu1,xiaoqu1=xiaoqu(shizhong,'市中')
maxxiaoqu2,xiaoqu2=xiaoqu(huaiyin,'槐荫')
maxxiaoqu3,xiaoqu3=xiaoqu(lixia,'历下')
maxxiaoqu4,xiaoqu4=xiaoqu(licheng,'历城')
maxxiaoqu5,xiaoqu5=xiaoqu(gaoxin,'高新')
maxxiaoqu6,xiaoqu6=xiaoqu(tianqiao,'天桥')
maxxiaoqu7,xiaoqu7=xiaoqu(changqing,'长清')
maxxiaoqu8,xiaoqu8=xiaoqu(zhangqiu,'章丘')
#画各个行政区的二手房最贵房价
plt.cla()
plt.bar([qu[0]+'区'+xiaoqu1,qu[1]+'区'+xiaoqu2,qu[2]+'区'+xiaoqu3,qu[3]+'区'+xiaoqu4,qu[4]+'区'+xiaoqu5,qu[5]+'区'+xiaoqu6,qu[6]+'区'+xiaoqu7,qu[7]+'区'+xiaoqu8]
,[maxxiaoqu1,maxxiaoqu2,maxxiaoqu3,maxxiaoqu4,maxxiaoqu5,maxxiaoqu6,maxxiaoqu7,maxxiaoqu8],color='g')
plt.title('各个行政区的二手房最贵房价')
plt.savefig(path+'各个行政区的二手房最贵房价.png')


'''
方法名称：  mj_zj
功能：      画出面积与总价的散点图，并建立一元线性回归模型，利用面积估算该区房价，并画到散点图上
参数：        df    数据
              s    区名
返回：        jia   面积为平均面积时的预测总价
'''
def mj_zj(df,s):
    plt.cla()#清空画板
    df=df[df['面积']<=300]#去除丽群孤立数据
    df=df[df['总价']<=1000]
    x=df['面积']
    y=df['总价']
    plt.scatter(x,y, label='数据点', color='b', s=25, marker="o")#做出散点图
    plt.xlabel('x面积（平方米）')
    plt.ylabel('y总价（万元）')
    plt.title(s+'\n建筑面积和总价的关系')
    
    area = df[["面积"]]
    price = df[["总价"]]
    #使用线性回归拟合
    linear = LinearRegression()
    #训练
    model = linear.fit(area,price)
    print(model.intercept_,model.coef_)#打印截距和回归系数
    price_ = model.predict(area)#预测(根据area预测得到的总价)
    plt.plot(area,price_,color="red",label='根据平均面积预测的房价')
    #plt.text(35,500, r'红线为根据面积预测的房价')
    plt.legend()
    plt.savefig(path+'建筑面积和总价的关系_'+s+'.png')
    jia=model.predict(pingjunmianji)
    return jia[0][0]

yucefangjia=[mj_zj(shizhong,'市中'),
mj_zj(huaiyin,'槐荫'),
mj_zj(lixia,'历下'),
mj_zj(licheng,'历城'),
mj_zj(gaoxin,'高新'),
mj_zj(tianqiao,'天桥'),
mj_zj(changqing,'长清'),
mj_zj(zhangqiu,'章丘')]#根据面积为平均面积时得到的各区预测房价

q1=shizhong[shizhong['面积']==round(pingjunmianji-1)]
q2=huaiyin[huaiyin['面积']==round(pingjunmianji-1)]
q3=lixia[lixia['面积']==round(pingjunmianji-1)]
q4=licheng[licheng['面积']==round(pingjunmianji-1)]
q5=gaoxin[gaoxin['面积']==round(pingjunmianji-1)]
q6=tianqiao[tianqiao['面积']==round(pingjunmianji-1)]
q7=changqing[changqing['面积']==round(pingjunmianji-1)]
q8=zhangqiu[zhangqiu['面积']==round(pingjunmianji-1)]
miji_107=[sum(q1['总价'])/len(q1),
          sum(q2['总价'])/len(q2),
          sum(q3['总价'])/len(q3),
          sum(q4['总价'])/len(q4),
          sum(q5['总价'])/len(q5),
          sum(q6['总价'])/len(q6),
          sum(q7['总价'])/len(q7),
          sum(q8['总价'])/len(q8)]
#做平均房价与预测房价的直方图
plt.cla()
total_width,n= 0.50,3
x =list(range(len(qu)))
width = total_width / n
plt.bar(x,zongjia,label="平均房价",tick_label = qu, color='g',width=width)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, yucefangjia, width=width, label='预测房价',fc = 'r')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, miji_107, width=width, label='面积为107时平均房价',fc = 'b')
plt.title('根据面积为平均面积(107.7)时得到的各区预测房价与平均房价')
plt.legend()
plt.savefig(path+'各个行政区的二手房平均房价与预测房价.png')

'''
方法名称：  dyhg
功能：     划分测试集和训练集，建立多元线性回归模型，画出实际的房价与预测房价比较图，并返回score
参数：        df    数据
              s    区名
返回：        score   模型精确度
'''
def dyhg(df,s):
    try:
        del df['小区名称']
    except:
        print('删除')
    
    cols = ["面积","室","厅","东","东北","东南","南","西","西北","西南","低层","高层","建楼时间"]
    X = df[cols]
    X.head()
    Y = df["总价"]
    Y.head()
    # 划分测试集和训练集
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=123)#对数据集按8:2比例划分为训练集和测试集
    
    #多元线性回归
    linear_multi = LinearRegression()
    model = linear_multi.fit(x_train,y_train)#训练得到模型
    print(model.intercept_,model.coef_)#截距，每个特征项的回归系数
    #多元线性回归的截距和回归系数
    predict_result = model.predict(x_test)#根据测试集得到预测结果
    score = model.score(x_test,y_test)#对 model 用 R^2 的方式进行打分，输出精确度。
    print('R-scores:',score)
    plt.cla()#清空画板
    plt.plot(range(len(x_test)),y_test,color="g",label='测试集中实际的房价')
    plt.plot(range(len(x_test)),predict_result,color="red",label='模型预测的房价')
    
    plt.title(s+'\n测试集中实际的房价与预测房价')
    plt.legend()
    plt.savefig(path+'测试集中实际的房价与预测房价_'+s+'.png')
    return score
R=[
dyhg(shizhong,'市中'),
dyhg(huaiyin,'槐荫'),
dyhg(lixia,'历下'),
dyhg(licheng,'历城'),
dyhg(gaoxin,'高新'),
dyhg(tianqiao,'天桥'),
dyhg(changqing,'长清'),
dyhg(zhangqiu,'章丘')]

plt.cla()#清空画板
plt.bar(qu,R,color="r",label='R-scores')
plt.title('各个行政区的R-scores')
plt.legend()
plt.savefig(path+'各个行政区的R-scores.png')

##使用多种特征的组合都可以预测房价，但是那种组合是最佳的组合喃，我们需要使用假设检验法，选出最佳的特征
#使用假设检验法

"""
方法名称：  jsyz
功能：      打印出假设检验的系列信息
参数：        df    数据
"""
def jsyz(df):
    Y = df["总价"].values
    cols = ["面积","室","厅","东","东北","东南","南","西","西北","西南","低层","高层","建楼时间"]
    X = df[cols]
    X_ = sm.add_constant(X)#将X增加为一个常量const（值为1.0）到X_
    #使用最小平方法计算性能
    result = sm.OLS(Y,X_)
    #fit方法真正进行运行计算
    summary = result.fit()
    #调用summary2方法，打印出假设检验的系列信息
    print(summary.summary2())

"""
名词解释
R-squared: 拟合度：用 R^2 的方式进行打分，精确度。越接近1越好（主要指标）
AIC：是衡量统计模型拟合优良性(Goodness of fit)的一种标准。越小越好（主要指标）

Coef 回归系数
Std.Err 标准差
t 虚无假设成立时的t值
P>|t| 虚无假设成立时的概率值
[0.025 ,0.975] 97.5%置信估计
要做假设性检验，首先要设置显著性标准。a.假设显著性标准是0.01 b.推翻虚无假设的标准是 p < 0.01 c.上面的SqFt的t=9.2416,P（>t） = 0.0000 < 0.01,因此虚无假设被推翻（这里的虚无假设是SqFt对price的回归系数为0，即SqFt与price不相关）
F统计
回归平方和 Regression Square Sum [RSS] :依变量的变化归咎于回归模型 A = sum((y-y*)^2)
误差平方和 Error Square Sum [ESS] : 依变量的变化归咎于线性模型 B = sum((y-y')^2)
总的平方和 Total Square Sum [TSS] : 依变量整体变化 C = A+B
回归平方平均 Model Mean Square: =RSS/Regression d.f(k) k=自变数的数量
误差平方平均 Error Mean Square:= ESS / Error d.f(n-k-1) n=观测值得数量
F统计 F = Model Mean Square / Error Mean Square
F值越大越好，Prob(F-statistic)越小越好

R Square：回归可以解释的变异比例，可以作为自变量预测因变量准确度的指标
SSE （残差平方和） = sum((y-y')^2)
SST （整体平方和） = sum((yi-yavg)^2)
R^2 = 1-SSE/SST 一般要大于0.6,0.7才算好
Adjust R Square
R^2 = 1-SSE/SST SSE最小，推导出R^2不会递减
yi = b1x1 + b2x2 + .... bkxk + .... 增加任何一个变量还会增加R^2
Adj R^2 = 1-(1-R^2)*((n-1)/(n-p-1))
n为总体大小，p为回归因子个数
AIC/BIC

AIC：（The Akaike Information Criterion）= 2K + nln(SSE/n) k是参数数量，n是观察数，SSE是残差平方和。 AIC鼓励数据拟合的优良性，但是尽量避免出现过拟合，所以优先考虑的模型应该是AIC最小的那一个，赤池信息量的准则是寻找可以最好的解释数据但是包含最少自由参数的模型
BIC (The Bayesain Information Criterion)
"""
#打印出各个行政区的假设检验的系列信息
jsyz(shizhong)
jsyz(huaiyin)
jsyz(lixia)
jsyz(licheng)
jsyz(gaoxin)
jsyz(tianqiao)
jsyz(changqing)
jsyz(zhangqiu)
"""
方法名称：  xzminaic
功能：      寻找AIC最小的属性作为预测的特征属性，并带入模型求出score
参数：        df    数据
返回：       score     模型精确度
             cols       特征属性
"""
#使用AIC，找出AIC最小的属性作为预测的特征属性
#寻找最小AIC的属性组合
def xzminaic(df):
    Y = df["总价"].values
    fileds = ["面积","室","厅","东","东北","东南","南","西","西北","西南","低层","高层","建楼时间"]
    acis = {}#字典
    for i in range(1,len(fileds)+1):
        for virables in itertools.combinations(fileds,i):#随机组合fileds里的信息，返回
            x1 = sm.add_constant(df[list(virables)])
            x2 = sm.OLS(Y,x1)
            res = x2.fit()
            acis[virables] = res.aic

    #使用collections里面的Counter，对字典进行统计
    counter = Counter(acis)
    #倒序取出最后10个特征组合
    counter.most_common()[::-1]
    
    
    cols = list(counter.most_common()[::-10][0][0])
    X = df[cols]
    X.head()
    Y = df["总价"]
    Y.head()
    # 划分测试集和训练集
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=123)#对数据集按8:2比例划分为训练集和测试集
    
    #多元线性回归
    linear_multi = LinearRegression()
    model = linear_multi.fit(x_train,y_train)#训练得到模型
    print(model.intercept_,model.coef_)#截距，每个特征项的回归系数
    #多元线性回归的截距和回归系数
    predict_result = model.predict(x_test)#根据测试集得到预测结果
    score = model.score(x_test,y_test)#对 model 用 R^2 的方式进行打分，输出精确度。
    print('R-scores:',score)
    return score,cols

s0,c0=xzminaic(shizhong)
s1,c1=xzminaic(huaiyin)
s2,c2=xzminaic(lixia)
s3,c3=xzminaic(licheng)
s4,c4=xzminaic(gaoxin)
s5,c5=xzminaic(tianqiao)
s6,c6=xzminaic(changqing)
s7,c7=xzminaic(zhangqiu)
R2=[s0,s1,s2,s3,s4,s5,s6,s7]


plt.cla()
total_width,n= 0.50,2
x =list(range(len(qu)))
width = total_width / n
plt.bar(x,R,label="全部属性组合",tick_label = qu, color='g',width=width)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, R2, width=width, label='最小AIC的属性组合',fc = 'r')
plt.title('各区全部属性组合与最小AIC的属性组合的R-scores')
plt.legend()
plt.savefig(path+'各区全部属性组合与最小AIC的属性组合的R-scores.png')

zuhe=[c0,c1,c2,c3,c4,c5,c6,c7]
print('行政区'+'---'+'R-scores'+'---'+'预测所需属性组合')
for i in range(8):
    print(qu[i]+'  ---'+'%.6f' %(R2[i])+'---'+str(zuhe[i]))