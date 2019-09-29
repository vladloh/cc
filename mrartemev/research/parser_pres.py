import dbworker
from multiprocessing import Pool, Process
from queue import Queue
import json
import time
from sconfig import size
from tb import bot
import random

from vk_parser import get_last_vk
from i_check import get_last_inst
from dbworker import  insert_2
def get_last_twitter(id, last_time):
    return 'twittermem'


def get_last_time(network, id):
    res = dbworker.get_current_state(id, network)
    return res

def filter_posts(posts, last_time):
    #print(posts, last_time)
    res = [i for i in posts if i['time'] > last_time]
    return res


get_last_posts = {'vk' : get_last_vk, 'inst' : get_last_inst, 'tw' : get_last_twitter}

def get_actual_posts(item):
    #print('item', item)
    id = item['id']
    network = item['network']
    last_time = get_last_time(network, id)
    #print(last_time)
    try:
        posts =  get_last_posts[network](id)#format: {url, text, network, time, id}
        #print('posts', posts)
        posts = filter_posts(posts, last_time)
        return posts
    except:
        return []


    
def iq300split(data, size = size):
    buff = []
    res = []
    for i in data:
        if len(buff) < size:
            buff.append(i)
        if len(buff) == size:
            res.append(buff)
            buff = []
    if len(buff): 
        res.append(buff)
    return res


def merge(kek):
    mem = []
    for i in kek:
        if i:
            for j in i:
                mem.append(j)
    return mem

def kek(portion):
    #print(portion)
    with Pool(size) as p:
        result = p.map(get_actual_posts, portion)
    for item in result:
        if item:
            mx = max(item, key = lambda i : i['time'])
            #print(mx, end = '\n\n')
            dbworker.set_state(mx['id'], mx['time'], mx['network'])
    return merge(result)

def add_posts(posts):
    print(f"len = {len(posts)}")
    for i in posts:
        q.put(i)

q = Queue()

def get_rating(post):
    text = post['text']
    return random.uniform(0, 1)


def process(): 
    time.sleep(6)
    sz = q.qsize()
    data = []
    for i in range(sz):
        getted = q.get()
        data.append(getted)
    data = sorted(data, key = lambda i: get_rating(i), reverse = True)
    #print(len(data))
    for item in data:
        try:
            text = item['text']
            url = item['url']
            js = json.dumps(item)
            insert_2(js)
        except:
            pass
        users = dbworker.get_all_users()
        for i in users:
            try:
                bot.send_message(i, text = url)
            except:
                pass
        
if __name__ == "__main__":
    #p = Process(target=process)
    #p.start()
    cnt = 0
    limit = 5
    while True:
        print(f"Step {cnt}")
        cnt += 1
        with open('accs_pres.json', 'r') as file:
            data = json.load(file)
        res = []
        for i in data:
            for j in data[i][:limit]:
                res.append({'network' : i, 'id' : j})
        #print(res)
        splitted_data = iq300split(res)
        for item in splitted_data:
            posts = kek(item)
            add_posts(posts)
            print(f"sz = {q.qsize()}")
        process()
        
    
        


    