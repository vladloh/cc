from instagram import Account, Media, WebAgent
import json
import time



def mem(account = "ccacc_ount"):
    last = None
    agent = WebAgent()
    account = Account(account)
    media1 = agent.get_media(account, count=1)
    tm = time.time()
    while (1):
        if (time.time() > tm + 5):
            media1 = agent.get_media(account)
            tm = time.time()
            if last != media1[0].code:
                last = media1[0].code
                print(last)

if __name__ == "__main__":
    mem()