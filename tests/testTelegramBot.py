import asyncio

from model.IA.TelegramBot.Bot import TradingBot

if __name__ == "__main__":
    bot = TradingBot("6134566086:AAHng2f_RwYvdaCCCz7CRNw9ePuIg8T5hVU", "5525135999", "DarkSpiderX")
    bot.start()