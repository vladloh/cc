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
TB2_NAME = "vlad_sosi"


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
def get_all_1(cursor):
    cursor.execute('''
        SELECT * FROM {};
        '''.format(TABLE_NAME))
    records = cursor.fetchall()
    return records


@command
def reset_table_1(cursor):
    cursor.execute('''
        DROP TABLE IF EXISTS {};
        '''.format(TABLE_NAME))
    cursor.execute('''
        CREATE TABLE {} (
        ID INTEGER NOT NULL PRIMARY KEY,
        TELEGRAM_ID INTEGER NOT NULL );
                   '''.format(TABLE_NAME))


@command
def reset_table_2(cursor):
    cursor.execute('''
        DROP TABLE IF EXISTS {};
        '''.format(TB2_NAME))
    cursor.execute('''
        CREATE TABLE {} (
        ID INTEGER NOT NULL PRIMARY KEY,
        POST STRING NOT NULL);
                   '''.format(TB2_NAME))


@command
def insert_1(cursor, uid):
    cursor.execute('''
        INSERT INTO {} (TELEGRAM_ID)
        VALUES (?);
    '''.format(TABLE_NAME, ), (uid, ))

@command
def insert_2(cursor, post):
    cursor.execute('''
        INSERT INTO {} (POST)
        VALUES (?);
    '''.format(TB2_NAME, ), (post, ))

@command
def delete_1(cursor, uid):
    cursor.execute('''
        DELETE FROM {} WHERE TELEGRAM_ID = ?;
    '''.format(TABLE_NAME, ), (uid, ))

@command
def delete_2(cursor, post):
    cursor.execute('''
        DELETE FROM {} WHERE POST = ?;
    '''.format(TB2_NAME, ), (post, ))

@command
def delete_all_1(cursor):
    cursor.execute('''
            DELETE FROM {};
            '''.format(TABLE_NAME))

@command
def delete_all_2(cursor):
    cursor.execute('''
            DELETE FROM {};
            '''.format(TB2_NAME))

def insert_user(telegram_id):
    insert_1(telegram_id)

def insert_post(post):
    insert_2(post)

def get_all_users():
    return list(set([j for i, j in get_all_1()]))

def get_all_posts():
    return list(set([json.loads(j) for i, j in get_all_2()]))

if __name__ == "__main__":
    reset_table_1()
    reset_table_2()