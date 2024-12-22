import os

from Bot import FAQBot
from train import train


# def setup_environment():
#     """
#     Настройка всего окружения: NLTK компонентов и создание модели
#     """
#     print("Начинаем настройку окружения...")
#
#     # Настройка SSL для NLTK
#     try:
#         _create_unverified_https_context = ssl._create_unverified_context
#     except AttributeError:
#         pass
#     else:
#         ssl._create_default_https_context = _create_unverified_https_context
#
#     # Создаем директорию для данных NLTK
#     nltk_data_dir = os.path.expanduser('~/nltk_data')
#     if not os.path.exists(nltk_data_dir):
#         os.makedirs(nltk_data_dir)
#
#     # Загружаем все необходимые компоненты NLTK
#     print("Загружаем компоненты NLTK...")
#     required_nltk_data = ['punkt', 'stopwords', 'punkt_tab']
#     for item in required_nltk_data:
#         try:
#             nltk.download(item, quiet=True)
#             print(f"Успешно загружен {item}")
#         except Exception as e:
#             print(f"Ошибка при загрузке {item}: {str(e)}")

def run_interactive_bot():
    """
    Запуск интерактивного бота с проверками
    """
    try:
        # Проверяем наличие файла модели
        if not os.path.exists('faq_bot.pth'):
            print("Файл модели не найден. Запустите сначала setup.py для создания модели.")
            return

        print("Загрузка модели...")
        model = FAQBot()
        model.load_model('faq_bot.pth')
        print("Модель загружена успешно")

        print("\nЗадавайте вопросы (для выхода введите 'выход' или 'exit')")

        while True:
            # Получаем вопрос от пользователя
            question = input("\nВаш вопрос: ").strip()

            # Проверяем условие выхода
            if question.lower() in ['выход', 'exit', 'quit', 'q']:
                print("До свидания!")
                break

            # Пропускаем пустые вопросы
            if not question:
                print("Вопрос не может быть пустым")
                continue

            try:
                # Получаем ответ от модели
                answers = model.find_answer(question, top_k=1)

                # Выводим результат
                for answer, confidence in answers:
                    print(f"\nОтвет (уверенность: {confidence:.2%}):")
                    print(answer)

                    # Предупреждение при низкой уверенности
                    if confidence < 0.5:
                        print("\nПримечание: уверенность в ответе низкая, возможно, стоит переформулировать вопрос")

            except Exception as e:
                print(f"Ошибка при получении ответа: {str(e)}")

    except Exception as e:
        print(f"Ошибка при запуске бота: {str(e)}")


if __name__ == "__main__":
    if not os.path.exists('faq_bot.pth'):
        # setup_environment()
        train()
    run_interactive_bot()
