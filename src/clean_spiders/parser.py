
import time

def get_last_time(network, id):
    return time.time() - 60 * 60

def get_last_vk(id, last_time):
    return 'vkmem'
def get_last_inst(id, last_time):
    return 'instmem'
def get_last_twitter(id, last_time):
    return 'twittermem'

get_last_posts = {'vk' : get_last_vk, 'inst' : get_last_inst, 'twitter' : get_last_twitter}

def get_actual_posts(network, id):
    last_time = get_last_time(network, id)
    posts =  get_last_posts[network]()#format: {url, text, network}
    print(posts)

if __name__ == "__main__":
    get_actual_posts('vk', 'id1')