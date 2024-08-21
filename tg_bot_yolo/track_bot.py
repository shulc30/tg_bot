from telegram.ext import Application, CommandHandler, MessageHandler, filters
import cv2
import os
from dotenv import load_dotenv
from ultralytics import YOLO  # Подключаем модель YOLO

load_dotenv()
TOKEN = os.environ.get("TOKEN")

# Инициализация бота
application = Application.builder().token(TOKEN).build()

# Загрузка модели YOLO
model = YOLO("yolov8n.pt")

async def start(update, context):
    await update.message.reply_text('Привет! Пришлите видео, и я обработаю его.')

async def process_video(update, context):
    video = await update.message.video.get_file()
    video_name = video.file_path.split("/")[-1]
    await video.download_to_drive(video_name)

    # Открываем видео
    cap = cv2.VideoCapture(video_name)
    
    # Создаем объект для записи видео с использованием кодека 'mp4v'
    output_video_name = f"processed_{video_name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
      ret, frame = cap.read()
      if not ret:
        break
    
    # Применяем модель YOLO для обнаружения объектов
      results = model(frame)
    
    # Рисуем рамки вокруг обнаруженных объектов
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

    # Отправляем обработанное видео обратно
    await update.message.reply_video(video=open(output_video_name, 'rb'))

    # Удаляем временные файлы
    os.remove(video_name)
    os.remove(output_video_name)

# Добавляем обработчики команд
application.add_handler(CommandHandler("start", start))
print('Бот запущен...')
application.add_handler(MessageHandler(filters.VIDEO, process_video))

# Запуск бота
if __name__ == "__main__":
    application.run_polling()




     

