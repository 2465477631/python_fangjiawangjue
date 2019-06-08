#重写getHouseInfo将info字典中的数据保存到csv文件中
#使用pandas，将字典转化成字段数组
import pandas as pd
import time
import requests as req
from bs4 import BeautifulSoup
import re
import csv
import multiprocessing
#头
headers ={'user-Agent':'Mozilla/5.0 Chrome/46.0.2490.80'}
#网站示例：https://jn.esf.fang.com/house-a0387/i32/
domain = "https://jn.esf.fang.com"
#每页60个记录，
#市中：/house-a0384/--100槐荫：/house-a0387/--100历下：/house-a0386/--100
#历城：/house-a0388/--100高新：/house-a0643/--100天桥：/house-a0385/--100
#长清：/house-a0645/--41章丘：/house-a0649/--80
#以下区域数据太少排除在外
#济阳：/house-a0648/--商河：/house-a0646/--平阴：/house-a0647/--
#莱芜：/house-a014785/--钢城：/house-a014786/
city_l =["/house-a0384/","/house-a0387/","/house-a0386/","/house-a0388/","/house-a0643/","/house-a0385/","/house-a0645/","/house-a0649/"]

'''
方法名称：pageFun1
功能：    爬取数据
参数：      cs     属于哪一个区          
''' 
def pageFun1(cs):
    df = pd.DataFrame()#定义函数返回数据
    time.sleep(cs)
    page_info_list=[]
    for i in range(1,100):
        try:
            page_url = domain + city_l[cs] +"i3"+ str(i)
            print(page_url)
            res = req.get(page_url,headers=headers)
            time.sleep(1)
            # houses =  soup.xpath(' /html/body/div[3]/div[1]/div[4]/div[5]/dl')
            # /html/body/div[3]/div[1]/div[4]/div[5]/dl
            
            if res.status_code ==200:
                soup = BeautifulSoup(res.text,"html.parser")#内置，或用"html5lib"
                houses = soup.select(".shop_list dl")
                for house in houses:
                    #加try except异常处理
                    try:
                        #  biaoti=house.find_all('span', class_="tit_shop")[0].string 
                        #例如： 发祥巷 近火车站 省立医院 西市场 双气 两室朝阳没遮挡 标题（没用）
                        l=house.find_all('p', class_="tel_shop")[0].get_text().split()
                        leixing=l[0]
                        #例如：2室1厅 类型
                        pattern = re.compile(r'\d+')   # 正则表达式：查找数字
                        mianji=pattern.findall(l[1])[0]
                        #例如：91 面积
                        louceng=l[2].split('|')[-1]
                        #例如：低层（共26层）楼层
                        chaoxiang=l[3].split('|')[-1]
                        #例如：南向 朝向
                        shijian=pattern.findall(l[4])[0]
                        #例如：2010 建楼时间
                        zhongjia= house.find_all('span', class_="red")[0].get_text()
                        #例如：165万 总价
                        danjai=pattern.findall(house.find_all('span')[-1].get_text())[0]
                        #例如：18132 单价（元/m2）
                        xiaoqu=house.find_all('p',class_="add_shop")[0].get_text().split()[0]
                        #例如：发祥巷 小区名称
                        # dizhi=house.find_all('p',class_="add_shop")[0].get_text().split()[1]
                        #例如：西市场-经二路251号 小区位置（没用）
                        lists=[leixing,mianji,louceng,chaoxiang,shijian,zhongjia,danjai,xiaoqu]
                        #save_csv(lists,cs)
                        if(cs==0):
                            try:
                                with open('jinan_sz.csv', 'a', encoding='utf-8-sig', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow(lists)
                            except:
                                print('write csv error!')
                        elif(cs==1):
                            try:
                                with open('jinan_hy.csv', 'a', encoding='utf-8-sig', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow(lists)
                            except:
                                print('write csv error!')
                        elif(cs==2):
                            try:
                                with open('jinan_lx.csv', 'a', encoding='utf-8-sig', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow(lists)
                            except:
                                print('write csv error!')
                        elif(cs==3):
                            try:
                                with open('jinan_lc.csv', 'a', encoding='utf-8-sig', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow(lists)
                            except:
                                print('write csv error!')
                        elif(cs==4):
                            try:
                                with open('jinan_gx.csv', 'a', encoding='utf-8-sig', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow(lists)
                            except:
                                print('write csv error!')
                        elif(cs==5):
                            try:
                                with open('jinan_tq.csv', 'a', encoding='utf-8-sig', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow(lists)
                            except:
                                print('write csv error!')
                        elif(cs==6):
                            try:
                                with open('jinan_cq.csv', 'a', encoding='utf-8-sig', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow(lists)
                            except:
                                print('write csv error!')
                        elif(cs==7):
                            try:
                                with open('jinan_zq.csv', 'a', encoding='utf-8-sig', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow(lists)
                            except:
                                print('write csv error!')
                        page_info_list.append(lists)
                    except Exception as e:
                        print("---------->",e)
            else:
                continue
        except Exception as e:
            print("Exception :",e)
    df = pd.DataFrame(page_info_list)
    return df

'''
方法名称：save_csv
功能：    将数据按行储存到csv文件中
参数：        house_data    获取到的房屋数据列表
            i               属于哪一个区
'''   
def save_csv(house_data,i):
    if(i==0):
        
        try:
            with open('jinan_sz.csv', 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(house_data)
        except:
            print('write csv error!')
    elif(i==1):
        try:
            with open('jinan_hy.csv', 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(house_data)
        except:
            print('write csv error!')
    elif(i==2):
        try:
            with open('jinan_lx.csv', 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(house_data)
        except:
            print('write csv error!')
    elif(i==3):
        try:
            with open('jinan_lc.csv', 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(house_data)
        except:
            print('write csv error!')
    elif(i==4):
        try:
            with open('jinan_gx.csv', 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(house_data)
        except:
            print('write csv error!')
    elif(i==5):
        try:
            with open('jinan_tq.csv', 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(house_data)
        except:
            print('write csv error!')
    elif(i==6):
        try:
            with open('jinan_cq.csv', 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(house_data)
        except:
            print('write csv error!')
    elif(i==7):
        try:
            with open('jinan_zq.csv', 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(house_data)
        except:
            print('write csv error!')   
#
#利用多线程的方式同时爬取8个区的住房数据
if __name__ == '__main__':
    
    for i in range(8):
        save_csv(['类型','面积','楼层','朝向','建楼时间','总价','单价（元/m2）','小区名称'],i)#设置表头
        p = multiprocessing.Process(target=pageFun1, args=(i,))
        p.start()
