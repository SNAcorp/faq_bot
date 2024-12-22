from Bot import FAQBot
from DataProc import DataProcessor


def train():
    try:
        # Инициализация обработчика данных
        print("Загрузка данных...")
        processor = DataProcessor('data.json')
        questions, _, answers, _ = processor.prepare_datasets(test_size=0.1)

        # Создание и инициализация модели
        model = FAQBot()

        # Построение базы знаний
        model.build_knowledge_base(questions, answers)

        # Сохранение модели
        model.save_model('faq_bot.pth')

        # Примеры использования
        test_questions = [
            "Сколько бюджетных мест на факультете?",
            "Как получить повышенную стипендию?",
            "Есть ли на факультете военная кафедра?",
            "Как организована практика студентов?",
            "Есть ли на факультете магистратура?"
        ]

        print("\nПримеры работы бота:")
        for question in test_questions:
            print(f"\nВопрос: {question}")
            answers = model.find_answer(question, top_k=1)
            for answer, confidence in answers:
                print(f"Уверенность: {confidence:.2%}")
                print(f"Ответ: {answer}")

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")