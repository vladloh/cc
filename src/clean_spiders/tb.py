import telebot
from sconfig import telegram_token
bot = telebot.TeleBot(telegram_token)




if __name__ == "__main__":
    bot.polling(none_stop=True)
    print('kek')