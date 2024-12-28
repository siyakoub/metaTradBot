from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CallbackQueryHandler, MessageHandler, filters, CommandHandler
import asyncio


class TradingBot:

    def __init__(self, bot_token: str, user_id: str, user_name: str):
        self.bot_token = bot_token
        self.user_id = user_id
        self.user_name = user_name
        self.bot = Bot(token=self.bot_token)
        # Créer l'application
        self.application = Application.builder().token(self.bot_token).build()

    async def initialize(self, update, context):
        # Créer le clavier avec les 4 options
        keyboard = [
            [
                KeyboardButton("Configurer mon MetaTrader 5 🔧"),
                KeyboardButton("Lancer des trade manuellement 📈")
            ],
            [
                KeyboardButton("Lancer le bot IA 🤖"),
                KeyboardButton("Récupérer les données d'un marché 📊")
            ]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

        # Envoyer le message de bienvenue
        await self.bot.send_message(
            chat_id=self.user_id,
            text=f'Salut {self.user_name}!👋🏽\n'
                 f'Je suis le bot développé par SKB pour t\'aider dans tes nouveaux projets de trading. 🤖💹\n'
                 f'Enchanté de faire ta connaissance ! 😇\n'
                 f'Alors... On fait quoi BOSS ? 😎',
            reply_markup=reply_markup
        )

    async def handle_message(self, update, context):
        # Récupérer le texte du message de l'utilisateur
        user_input = update.message.text

        # Traiter les différentes options du menu
        if user_input == 'Configurer mon MetaTrader 5 🔧':
            await update.message.reply_text(f"Tu as choisi l'option {user_input}!")
        elif user_input == 'Lancer des trade manuellement 📈':
            await update.message.reply_text(f"Tu as choisi l'option {user_input}!")
        elif user_input == 'Lancer le bot IA 🤖':
            await update.message.reply_text(f"Tu as choisi l'option {user_input}!")
        elif user_input == 'Récupérer les données d\'un marché 📊':
            await update.message.reply_text(f"Tu as choisi l'option {user_input}!")

    def start(self):
        self.application.add_handler(CommandHandler("start", self.initialize))
        # Ajouter un gestionnaire pour les messages texte
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        # Lancer l'application
        self.application.run_polling()