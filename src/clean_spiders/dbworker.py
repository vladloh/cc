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

import sqlite3


TABLE_NAME = "mem_table"


def command(func):
    def wrapper(*args, **kwargs):
        conn = sqlite3.connect('src/clean_spiders/my.db')
        try:
            res = func(conn.cursor(), *args, **kwargs)
        finally:
            conn.commit()
            conn.close()
        return res

    return wrapper


@command
def get_all(cursor):
    cursor.execute('''
        SELECT * FROM {};
        '''.format(TABLE_NAME))
    records = cursor.fetchall()
    return records


@command
def reset_table(cursor):
    cursor.execute('''
        DROP TABLE IF EXISTS {};
        '''.format(TABLE_NAME))
    cursor.execute('''
        CREATE TABLE {} (
        ID INTEGER NOT NULL PRIMARY KEY,
        TELEGRAM_ID INTEGER NOT NULL );
                   '''.format(TABLE_NAME))


@command
def insert(cursor, uid):
    cursor.execute('''
        INSERT INTO {} (TELEGRAM_ID)
        VALUES (?);
    '''.format(TABLE_NAME, ), (uid, ))


@command
def delete(cursor, uid):
    cursor.execute('''
        DELETE FROM {} WHERE TELEGRAM_ID = ?;
    '''.format(TABLE_NAME, ), (uid, ))


@command
def delete_all(cursor):
    cursor.execute('''
            DELETE FROM {};
            '''.format(TABLE_NAME))

def insert_user(telegram_id):
    insert(telegram_id)

def get_all_users():
    return list(set([j for i, j in get_all()]))

if __name__ == "__main__":
    reset_table()