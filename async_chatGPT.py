from telegram.ext import Application, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
import openai
import os

load_dotenv()

TOKEN = os.environ.get('TOKEN')
GPT_SECRET_KEY = os.environ.get('GPT_SECRET_KEY')

openai.api_key = GPT_SECRET_KEY

async def get_answer(text):
    completion = await openai.ChatCompletion.acreate(
        model = 'gpt-4o-mini',
        messages = [{'role': 'user', 'content': text}]
    )
    return completion.choices[0].message['content']

async def start(update, context):
    await update.message.reply_text('Задайте любой вопрос чат GPT.')

async def help_command(update, context):
    await update.message.reply_text('Вы можете пообщаться с chatGPT на любую тему.')

async def gpt(update, context):

    processing_message = await update.message.reply_text('Ваш вопрос обрабатывается...')

    res = await get_answer(update.message.text)

    await processing_message.delete()
    await update.message.reply_text(res)

def main():
    application = Application.builder().token(TOKEN).build()
    print('Бот запущен...')

    application.add_handler(CommandHandler('start', start, block=False))
    application.add_handler(CommandHandler('help', help_command, block=False))
    application.add_handler(MessageHandler(filters.TEXT, gpt, block=False))

    application.run_polling()

if __name__ == '__main__':
    main()

    