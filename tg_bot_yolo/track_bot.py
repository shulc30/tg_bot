from telegram.ext import Application, CommandHandler, MessageHandler, filters
import cv2
import time
import os
import asyncio
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()
TOKEN = os.environ.get("TOKEN")

# Инициализация бота
application = Application.builder().token(TOKEN).build()

# Загрузка модели YOLO
model = YOLO("yolov8n.pt")

# Глобальные параметры
settings = {
    "confidence": 0.5,  # Порог уверенности
    "classes": None  # Классы объектов (None означает все классы)
}

async def start(update, context):
    await update.message.reply_text('Привет! Пришлите видео, и я обработаю его.')

async def help_command(update, context):
    help_text = (
        "Я бот для трекинга объектов на видео!\n\n"
        "Вот как я могу вам помочь:\n"
        "1. Отправьте команду /start для начала работы.\n"
        "2. Пришлите мне видео, и я обработаю его, определяя объекты на видео.\n"
        "3. Я отправлю вам обработанное видео с выделенными объектами.\n"
        "4. Используйте команду /settings, чтобы настроить порог уверенности и классы объектов для трекинга.\n"
        "   Примеры классов:\n"
        "   - 0: Люди\n"
        "   - 1: Автомобили\n"
        "   - 2: Велосипеды\n"
        "   - 3: Мотоциклы\n"
        "   Пример: /settings 50 0 1 (Порог 50%, трекинг людей и автомобилей)\n"
        "5. Если у вас возникнут вопросы, используйте команду /help для получения этой информации.\n"
    )
    await update.message.reply_text(help_text)


async def settings_command(update, context):
    if context.args:
        if len(context.args) == 1 and context.args[0].isdigit():
            settings["confidence"] = float(context.args[0]) / 100
        elif len(context.args) > 1:
            settings["classes"] = list(map(int, context.args))
        await update.message.reply_text(f'Настройки обновлены: \nПорог уверенности: {settings["confidence"]}\nКлассы: {settings["classes"]}')
    else:
        await update.message.reply_text(f'Текущие настройки: \nПорог уверенности: {settings["confidence"]}\nКлассы: {settings["classes"]}')

async def process_video(update, context):
    video = await update.message.video.get_file()
    video_name = video.file_path.split("/")[-1]
    await video.download_to_drive(video_name)

    asyncio.create_task(process_and_return_video(update, context, video_name))

async def process_and_return_video(update, context, video_name):
    
    print(f"Используемые настройки: порог {settings['confidence']}, классы {settings['classes']}")
    time.sleep(5)
    cap = cv2.VideoCapture(video_name)
    
    output_video_name = f"processed_{video_name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Применяем модель YOLO с пользовательскими настройками
        results = model(frame, conf=settings["confidence"], classes=settings["classes"])
    
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    await update.message.reply_text('Видео обработано.')
    await update.message.reply_video(video=open(output_video_name, 'rb'))

    os.remove(video_name)
    os.remove(output_video_name)

# Добавляем обработчики команд
application.add_handler(CommandHandler("start", start))
print('Бот запущен...')
application.add_handler(CommandHandler("help", help_command))
application.add_handler(CommandHandler("settings", settings_command))
application.add_handler(MessageHandler(filters.VIDEO, process_video))

# Запуск бота
if __name__ == "__main__":
    application.run_polling()










     

