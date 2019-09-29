import telebot
from sconfig import telegram_token
from dbworker import insert_user

bot = telebot.TeleBot(telegram_token)
@bot.message_handler(commands = ['start'])
def add_user(message):
    bot.send_message(message.chat.id, 'hi')
    insert_user(message.chat.id)



if __name__ == "__main__":
    bot.polling(none_stop=True)