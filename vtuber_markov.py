# -*- coding: utf-8 -*-

import sys
import os
import chardet
import random
import MeCab
from pyjtalk.pyjtalk import PyJtalk

import concurrent.futures

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import *
import requests

from bs4 import BeautifulSoup
import csv

# 字幕データを入れるディレクトリ
# なかったら作成する
txt_path = './kizuna'
if not os.path.exists(txt_path):
        os.makedirs(txt_path)
re_txts = os.listdir(txt_path)#すでにある場合

#　動画番号
file_count =0;

#youtube 検索クラス
class Youtube():
    def __init__(self,_url,result=200):
        search_url = _url

        req = requests.get(search_url)
        soup = BeautifulSoup(req.text.encode(req.encoding).decode('utf-8','strict'),"html.parser")
        h3s = soup.find_all("h3", {"class":"yt-lockup-title"})[0:result+1]

        self.data = [h3 for h3 in h3s]
        self.url = ["https://www.youtube.com" + h3.a.get('href') for h3 in h3s]
        self.title = [h3.a.get("title") for h3 in h3s]
        self.id = [h3.a.get("href").split("=")[-1] for h3 in h3s]
        self.embed = ["https://www.youtube.com/embed/" + h3.a.get("href").split("=")[-1] for h3 in h3s]
        self.time = [h3.span.text.replace(" - 長さ: ","").replace("。","") for h3 in h3s]
        self.info = [h3.text for h3 in h3s] # >>タイトル　- 長さ：00:00。
    def select_all(self):
        values = {"url":self.url,"title":self.title,"id":self.id,"embed":self.embed,"time":self.time}
        info = self.info
        for i in range(len(info)):
            print("%s:%s" % (i,info[i]))
        results = {
            "url":values["url"],
            "title":values["title"],
            "id":values["id"],
            "embed":values["embed"],
            "time":values["time"],
            }
        return results

#マルコフ連鎖
class myMarkov:
    #_name
    wordlist=""
    wordlists=[]
    allwordlists=[]

    markov = {}
    w1 = ""
    w2 = ""

    count = 0
    sentence = ""

    first_word = True

    pn_count=0;
    all_pn_value=0;

    def __init__(self,name):
        self._name = name
    def wakati(self,text):#わかち書き
        t = MeCab.Tagger("-Owakati")
        m = t.parse(text)
        result = m.rstrip(" \n").split(" ")
        return result
    def make_table(self,file_name):#マルコフ連鎖テーブル作成
            txt = os.path.join(txt_path, file_name)
            src = open(txt,"rb").read()
            src = src.decode()

            #改行ごとに分かち書きする
            srcs = src.split()
            for src in srcs:
                self.wordlists.extend(self.wakati(src))
            self.allwordlists.extend(self.wordlists)
            #self.wordlist = self.wakati(src)
            for word in self.wordlists:
                if not word == '\u3000' and not word == 'quot':#除外
                    word.replace('\u3000', '')#一部が全角空白文字で文字化けしている単語を書き換え
                    if self.w1 and self.w2:
                        if(self.w1,self.w2) not in self.markov:
                            self.markov[(self.w1,self.w2)]=[]
                        #print('w1 がマルコフにないよ',self.w1)
                        #print('w2 がマルコフにないよ',self.w2)
                        self.markov[(self.w1,self.w2)].append(word) #要素のつながりをマルコフに入れとく
                    #print("w1,w2の組に「"+word+"」を追加する")
                    self.w1,self.w2 = self.w2,word
            self.wordlists=[]
    def show_table(self):#マルコフ辞書の中身を確認したい時に
        print(self.markov)
        #return self.markov
    def show_wordlist(self):
        print(self.allwordlists)
    def get_pn_value(self,word):#PNテーブルから該当の単語を検索して、極性値をとってくる。該当する単語が見当たらなかったら空白を返す
        diclist =[]
        src = open('pn_ja.dic',"rb").read()
        src = src.decode()
        for row in src.split("\n"):
            w = row.split("\t")[0]
            if w =="":
                break;
            else:
                if row.split(":")[0] == "":
                    break;
                d = {'Word':row.split(":")[0], 'Reading':row.split(":")[1], 'POS':row.split(":")[2], 'PN':float(row.split(":")[3])}
                diclist.append(d)
        values = [x['PN'] for x in diclist if x['Word'] == word]
        value = values[0] if len(values) else ''
        return value
    def generate(self):#セリフをを自動作成
        while self.first_word:#文の最初にきてはいけないもの（助詞、助動詞、句読点など）を除外する部分
            w1,w2 = random.choice(list(self.markov.keys()))#辞書の中からランダムにkeyを選ぶ
            tmp = random.choice(self.markov[(w1,w2)])#対応するvalueをランダムに選ぶ
            m = MeCab.Tagger("-Ochasen")
            keywords = m.parse(tmp)
            for row in keywords.split("\n"):
                w = row.split("\t")[0]
                if w =="EOS":
                    break;
                else:
                    pos = row.split("\t")[3].split("-")[0]
                    if not pos == "助詞" and not pos == "助動詞" and not tmp == "。" and not tmp == "ー" and not tmp == "…" and not tmp == "‥" and not tmp == "、"and not tmp == "？"and not tmp == "！":
                        self.first_word = False
                        break;
        while self.count < (random.randrange(30)+10):#マルコフ辞書から文を作成
            tmp = random.choice(self.markov[(w1,w2)])
            self.sentence +=tmp
            #極性チェック
            print(str(self.count+1)+"個目の極性チェック")
            print(tmp)

            if self.get_pn_value(tmp)!="":
                print(self.get_pn_value(tmp))
                self.all_pn_value+=float(self.get_pn_value(tmp))#極性を加算
                self.pn_count+=1;#極性のある単語を数える
            w1,w2 = w2,tmp#今度はw2 ,tmpに続くものの中から、ランダムに選ばれる
            self.count +=1#単語カウント

        print("生成文:"+self.sentence)
        print("極性ありのワード:"+str(self.pn_count))
        if self.pn_count>0:
            print("極性合計:"+str(self.all_pn_value))
            print("極性平均:"+str(self.all_pn_value/self.pn_count))

        return self.sentence



if __name__ == '__main__':

    #複数youtubeチャンネルのページから動画のIDをとってくる。
    Y = Youtube("https://www.youtube.com/channel/UC4YaOt1yT-ZeyB0OmxHgolA",result=200)
    Y2 = Youtube("https://www.youtube.com/channel/UCbFwe3COkDrbNsbMyGNCsDg",result=200)
    Y3 = Youtube("https://www.youtube.com/channel/UC4YaOt1yT-ZeyB0OmxHgolA/videos",result=200)

    ids =[]#動画のIDを格納する
    movie = Y.select_all()
    ids.append(movie["id"])
    movie = Y2.select_all()
    ids.append(movie["id"])
    movie = Y3.select_all()
    ids.append(movie["id"])

    ids_=[]
    for v in ids:#複数のチャンネルからとった動画のIDを一つに格納する
        for id in v:
            ids_.append(id)

    file_count=1
    line_format =""
    jimaku_count = 0
    for i in ids_:
        if len(str(i)) <12:#idが11文字以上だとプレイリストのIDなので除外
            r = requests.get('http://video.google.com/timedtext?type=list&v='+i)
            root = ET.fromstring(r.text)
            flg = False
            for child in root:
                str1 =str(child.attrib)
                if str1.find('ja')>1:#日本語字幕あり
                    flg =True
            if flg == True:
                if i.find('/channel/') == -1 and i.find('/user/') == -1 and len(str(i)) <12:#ユーザーとチャンネルのIDは除外する
                    jimaku_count+=1
                    url = 'http://video.google.com/timedtext?hl=ja&lang=ja&name=&v='+i
                    try:
                        r = requests.get(url)
                    except requests.exceptions.RequestException as err:
                        print(err)
                    root = ET.fromstring(r.text)
                    for child in root:
                        line_format+=child.text+"\n"
                    outfname = os.path.join('./kizuna',  "ai"+str(file_count)+".txt")
                    outfp = open(outfname, "w")
                    outfp.write(line_format)
                    file_count+=1
                    line_format =""
    print("サンプル動画全体:"+str(len(ids_)))
    print("うち字幕あり:"+str(jimaku_count))
    print("マルコフ連鎖を作るよ")

    my_markov = myMarkov("markov")#マルコフクラスを作成

    re_txts = os.listdir(txt_path)#とってきたディレクトリ内のデータを読み込む

    count=1#ファイルのカウント
    for txt in re_txts:#整形済みディレクトリにあるデータからマルコフテーブルを作る
        my_markov.make_table(txt)
        print(str(len(re_txts))+"個中"+str(count)+"個終了")
        count+=1

    #my_markov.show_table();
    #my_markov.show_wordlist()
    print("適当にセリフを生成します")
    p = PyJtalk()
    p.say(str(my_markov.generate()))#しゃべる。
