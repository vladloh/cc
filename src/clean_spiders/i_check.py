from instagram import Account, Media, WebAgent
import json
import time
import sconfig
import dbworker

def mem(account = sconfig.our, delay = sconfig.delay):
    last = None
    agent = WebAgent()
    account = Account(account)
    media1 = agent.get_media(account, count=1)
    tm = time.time()
    while (1):
        if (time.time() > tm + delay):
            media1 = agent.get_media(account)
            tm = time.time()
            if last != media1[0].code:
                last = media1[0].code
                print(last)

def get_last_inst(account = sconfig.our, cnt = 5):
    result = []
    agent = WebAgent()
    account = Account(account)
    media1 = agent.get_media(account, count=cnt)
    for i in media1[0]:
        result.append({'url': 'https://www.instagram.com/p/' + i.code + '/', 'time': i.date, 'text': i.caption, 'network': 'inst', 'id': i.owner})
    return result

hobbba = get_last_inst()

#if __name__ == "__main__":
    #mem()