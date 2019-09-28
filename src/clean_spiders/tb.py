import telebot
from telebot import types
from telebot import apihelper
from sconfig import telegram_token







#apihelper.proxy = {'https':'socks5://sox.ctf.su:1080', 'http' : 'socks5://sox.ctf.su:1080'}
bot = telebot.TeleBot(telegram_token)

bot.send_message(841622311, 'hi')