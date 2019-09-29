import telebot
from sconfig import telegram_token
from dbworker import insert_user, get_all_users

bot = telebot.TeleBot(telegram_token)
@bot.message_handler(commands = ['start'])
def add_user(message):
    users = get_all_users()
    if message.chat.id in users:
        bot.send_message(message.chat.id, 'Hi my dear friend')
    else:
        bot.send_message(message.chat.id, 'hello my new friend')
        insert_user(message.chat.id)



if __name__ == "__main__":
    bot.polling(none_stop=True)