import telebot
from sconfig import telegram_token
from dbworker import insert_user, get_all_users, get_all_posts

bot = telebot.TeleBot(telegram_token)

@bot.message_handler(commands = ['start'])
def add_user(message):
    users = get_all_users()
    if message.chat.id in users:
        bot.send_message(message.chat.id, 'Hi my dear friend')
    else:
        bot.send_message(message.chat.id, 'Hello my new friend')
        insert_user(message.chat.id)


@bot.message_handler(commands = ['new'])
#@bot.callback_query_handler(func = lambda call: True and call.data.startswith('new')
def get_post(message):
    res = get_all_posts()
    print(res)
    kek = res[0]
    bot.send_message(message.chat.id, kek['url'])

if __name__ == "__main__":
    bot.polling(none_stop=True)