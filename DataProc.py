import json
from typing import List, Tuple
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, json_path: str):
        """
        Инициализация обработчика данных

        Args:
            json_path (str): Путь к JSON файлу с FAQ данными
        """
        self.json_path = json_path
        self.data = None
        self.questions = None
        self.answers = None

    def load_data(self) -> None:
        """Загрузка данных из JSON файла"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            self.data = raw_data['data']

        self.questions = [item['question'] for item in self.data]
        self.answers = [item['answer'] for item in self.data]

    def prepare_datasets(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[
        List[str], List[str], List[str], List[str]]:
        """
        Разделение данных на тренировочную и тестовую выборки

        Args:
            test_size (float): Размер тестовой выборки (0-1)
            random_state (int): Seed для воспроизводимости результатов

        Returns:
            Tuple[List[str], List[str], List[str], List[str]]: train_questions, test_questions, train_answers, test_answers
        """
        if self.questions is None or self.answers is None:
            self.load_data()

        train_questions, test_questions, train_answers, test_answers = train_test_split(
            self.questions,
            self.answers,
            test_size=test_size,
            random_state=random_state
        )

        return train_questions, test_questions, train_answers, test_answers