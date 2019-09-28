from vedis import Vedis
import sconfig


#inst - инста, vk - выка, tw - твиттер

def get_current_state(user_id, social_network = 'inst'):
    with Vedis(sconfig.last_posts) as db:
        try:
            key = f'{social_network}_{user_id}'
            return db[key].decode()
        except KeyError: 
            return None 


def set_state(user_id, value, social_network = 'inst'):
    with Vedis(sconfig.last_posts) as db:
        try:
            key = f'{social_network}_{user_id}'
            db[key] = value
            return True
        except:
            print('Проснись, ты обосрался!') # Помянем Санька 
            return False

set_state("loshara", "hobahoba", 'inst')
print(get_current_state("loshara"))