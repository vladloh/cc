import dbworker
from multiprocessing import Pool, Process
from queue import Queue
import json
import time
from sconfig import size
from tb import bot

from vk_parser import get_last_vk
from i_check import get_last_inst
def get_last_twitter(id, last_time):
    return 'twittermem'





def get_last_time(network, id):
    res = dbworker.get_current_state(id, network)
    return res

def filter_posts(posts, last_time):
    res = [i for i in posts if i['time'] > last_time]
    return res


get_last_posts = {'vk' : get_last_vk, 'inst' : get_last_inst, 'tw' : get_last_twitter}

def get_actual_posts(item):
    id = item['id']
    network = item['network']
    last_time = get_last_time(network, id)
    #print(last_time)
    try:
        posts =  get_last_posts[network](id)#format: {url, text, network, time, id}
        posts = filter_posts(posts, last_time)
        return posts
    except:
        return None


    
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
def process(): 
    while True:
        time.sleep(3)
        sz = q.qsize()
        for i in range(sz):
            getted = q.get()
            print(getted['text'])
            bot.send_message(841622311, getted['text'])


if __name__ == "__main__":
    #p = Process(target=process)
    #p.start()
    cnt = 0
    while True:
        print(f"Step {cnt}")
        cnt += 1
        with open('src/clean_spiders/accs.json', 'r') as file:
            data = json.load(file)
        res = []
        for i in data:
            for j in data[i]:
                res.append({'network' : i, 'id' : j})
                
        splitted_data = iq300split(res)
        for item in splitted_data:
            posts = kek(item)
            add_posts(posts)
            print(f"sz = {q.qsize()}")
        process()
        

        


    