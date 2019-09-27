from instagram import Account, Media, WebAgent
import json



def mem():
    agent = WebAgent()
    account = Account("cosmopolitan_russia")

    media1, pointer = agent.get_media(account)
    count = 0 #счетчик колличества выполнений
    kek = [] #массив кеков
    for k in range(198):
        try:
            media2, pointer = agent.get_media(account, pointer=pointer, count=50, delay=0.4)
            for i in media2:
                kek.append({'text' : i.caption})
            count += 1
        except:
            pass
        print(1)
    dmp = json.dumps(kek)
    with open('mem.json', 'a') as f:
        print(dmp, file = f)


if __name__ == "__main__":
    mem()