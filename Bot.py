import re
from typing import List, Tuple

import nltk
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# Загружаем необходимые компоненты NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class FAQBot:
    def __init__(self, model_name: str = 'DeepPavlov/rubert-base-cased'):
        """
        Инициализация FAQ бота
        Args:
            model_name: название предобученной модели BERT
        """
        print(f"Загрузка модели {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Переводим модель в режим оценки и на CPU
        self.model.eval()
        self.model.cpu()

        # Инициализация базы знаний
        self.questions_embeddings = None
        self.answers = None
        self.original_questions = None

    def preprocess_text(self, text: str) -> str:
        """
        Предварительная обработка текста
        Args:
            text: входной текст
        Returns:
            обработанный текст
        """
        # Приведение к нижнему регистру
        text = text.lower()

        # Удаление пунктуации и лишних пробелов
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Получение эмбеддинга для текста
        Args:
            text: входной текст
        Returns:
            numpy array с эмбеддингом
        """
        # Предобработка текста
        text = self.preprocess_text(text)

        # Токенизация текста
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # Получение эмбеддинга
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Усредняем все токены последнего слоя для лучшего представления
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        return embedding

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Вычисление семантической близости между двумя эмбеддингами
        """
        # Reshape embeddings for sklearn's cosine_similarity
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)

        return cosine_similarity(emb1, emb2)[0][0]

    def build_knowledge_base(self, questions: List[str], answers: List[str]):
        """
        Построение базы знаний из вопросов и ответов
        Args:
            questions: список вопросов
            answers: список ответов
        """
        print("Создание базы знаний...")

        # Сохраняем оригинальные вопросы для анализа
        self.original_questions = questions

        # Получаем эмбеддинги для каждого вопроса
        embeddings = []
        for i, question in enumerate(questions):
            if i % 10 == 0:
                print(f"Обработано {i}/{len(questions)} вопросов")

            embedding = self.get_embedding(question)
            embeddings.append(embedding)

        self.questions_embeddings = np.stack(embeddings)
        self.answers = answers

        print("База знаний создана")

    def find_answer(self, question: str, top_k: int = 1, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Поиск ответа на вопрос
        Args:
            question: вопрос пользователя
            top_k: количество лучших ответов для возврата
            threshold: минимальный порог уверенности
        Returns:
            список кортежей (ответ, уверенность)
        """
        if self.questions_embeddings is None or self.answers is None:
            raise ValueError("База знаний не создана")

        # Получаем эмбеддинг вопроса
        question_embedding = self.get_embedding(question)

        # Вычисляем семантическую близость со всеми вопросами
        similarities = []
        for i in range(len(self.questions_embeddings)):
            similarity = self.calculate_similarity(question_embedding, self.questions_embeddings[i])
            similarities.append(similarity)

        similarities = np.array(similarities)

        # Находим top-k наиболее похожих ответов с учетом порога
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []

        for idx in top_indices:
            if similarities[idx] >= threshold:
                results.append((self.answers[idx], float(similarities[idx])))

            # Для отладки
            print(f"Debug - Похожий вопрос: {self.original_questions[idx]}")
            print(f"Debug - Сходство: {similarities[idx]:.4f}")

        # Если нет ответов с достаточной уверенностью
        if not results:
            return [("Извините, я не уверен в ответе на этот вопрос. Попробуйте переформулировать вопрос.", 0.0)]

        return results

    def save_model(self, path: str):
        """
        Сохранение базы знаний
        Args:
            path: путь для сохранения
        """
        torch.save({
            'questions_embeddings': self.questions_embeddings,
            'answers': self.answers,
            'original_questions': self.original_questions
        }, path)
        print(f"Модель сохранена в {path}")

    def load_model(self, path: str):
        """
        Загрузка базы знаний
        Args:
            path: путь к сохраненной модели
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.questions_embeddings = checkpoint['questions_embeddings']
        self.answers = checkpoint['answers']
        self.original_questions = checkpoint['original_questions']
        print(f"Модель загружена из {path}")