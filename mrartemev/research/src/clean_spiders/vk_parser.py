import vk_api
from sconfig import vk_token
import json

def get_last_vk(id, count = 5):
    vk_session = vk_api.VkApi(token = vk_token)
    res = vk_session.method(method = "wall.get", values = {'domain' : id, 'v' : '5.52', 'count' : count})
    items = res['items']
    data = []
    for i in items:
        try:
            nw = {}
            owner_id = i['owner_id']
            post_id = id
            nw['time'] = i['date']
            nw['text'] = i['text']
            nw['id'] = post_id
            nw['network'] = 'vk'
            nw['url'] = f'https://vk.com/wall{owner_id}_{post_id}'
            data.append(nw)
        except:
            pass
    return data


