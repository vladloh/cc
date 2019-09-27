from instagram import Account, Media, WebAgent
import json

agent = WebAgent()
account = Account("cosmopolitan_russia")

media1, pointer = agent.get_media(account)
count = 0 #счетчик колличества выполнений

for k in range(1):
    media2, pointer = agent.get_media(account, pointer=pointer, count=50, delay=0.3)
    kek = [] #массив кеков
    for i in media2:
        kek.append({'text' : i.caption})
    dmp = json.dumps(kek)
    with open('mem.json', 'a') as f:
        print(dmp, file = f)
    count += 1

print(count)