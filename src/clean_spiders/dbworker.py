from vedis import Vedis
import sconfig


#inst - инста, vk - выка, tw - твиттер

def get_current_state(user_id, social_network = 'inst'):
    with Vedis(sconfig.last_posts) as db:
        try:
            key = f'{social_network}_{user_id}'
            value =  int(db[key].decode())
            #print(f"get: key = {key}, value = {str(value)}")
            return value
        except KeyError: 
            value = 1368674341
            #print(f"get: key = {key}, value = {value}")
            return value 


def set_state(user_id, value, social_network = 'inst'):
    with Vedis(sconfig.last_posts) as db:
        try:
            key = f'{social_network}_{user_id}'
            db[key] = str(value)
            #print(f"add: key = {key}, value = {str(value)}")
            return True
        except:
            print('Проснись, ты обосрался!') # Помянем Санька 
            #print(key, str(value))
            return False