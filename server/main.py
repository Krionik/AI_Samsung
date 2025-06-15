import os
import sys
import threading
import time
import cv2
import dlib
import httpx
import numpy as np
import firebase_admin

from norm import FaceNormalizer
from scipy.spatial.distance import cosine
from config import config
from dotenv import load_dotenv
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import Transaction
from supabase import create_client
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple, List
from PyQt5.QtCore import QSize, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog)


class FaceRecognitionApp(QMainWindow):
    """Основное приложение для распознавания лиц с GUI и серверным режимом."""

    def __init__(self, is_server: bool = True):
        self.logger = config.logger.getChild('FaceRecognitionApp')
        self.is_server = is_server

        # Инициализация детектора и нормализатора
        self._init_face_components()

        # Загрузка эмбеддингов
        self._load_embeddings()

        if not self.is_server:
            super().__init__()
            self._init_gui()
        else:
            self._init_server_components()
            self._start_monitoring()

        self.logger.info("Приложение успешно инициализировано")

    def _init_face_components(self) -> None:
        """Инициализирует компоненты для работы с лицами."""
        self.logger.debug("Инициализация детектора и нормализатора...")
        self.face_detector = dlib.get_frontal_face_detector()
        self.normalizer = FaceNormalizer()

    def _load_embeddings(self) -> None:
        """Загружает предварительно сохраненные эмбеддинги."""
        try:
            self.logger.debug("Загрузка эмбеддингов...")
            data = np.load("./data/embeddings.npz")
            self.labels = data["labels"]
            self.embeddings = data["embeddings"]
            self.logger.info(f"Загружено {len(self.labels)} эмбеддингов")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки эмбеддингов: {str(e)}", exc_info=True)
            raise

    def _init_gui(self) -> None:
        """Инициализирует графический интерфейс."""
        self.logger.debug("Инициализация GUI...")
        self.setWindowTitle("Face Recognition App")
        self.setGeometry(100, 100, 800, 600)

        # Основные компоненты GUI
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Кнопки управления
        self._init_buttons()

        # Область отображения изображения
        self._init_image_display()

        # Настройки камеры
        self.camera = None
        self.camera_timer = QTimer()
        self.is_camera_active = False

    def _init_buttons(self) -> None:
        """Инициализирует кнопки управления."""
        self.button_layout = QHBoxLayout()

        self.select_btn = QPushButton("Select Image")
        self.select_btn.clicked.connect(self.open_image)
        self.button_layout.addWidget(self.select_btn)

        self.camera_btn = QPushButton("Start Camera")
        self.camera_btn.setEnabled(False)
        self.button_layout.addWidget(self.camera_btn)

        self.capture_btn = QPushButton("Capture Photo")
        self.capture_btn.setEnabled(False)
        self.button_layout.addWidget(self.capture_btn)

        self.layout.addLayout(self.button_layout)

    def _init_image_display(self) -> None:
        """Инициализирует область отображения изображения."""
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(QSize(1800, 900))
        self.layout.addWidget(self.image_label)

    def _init_server_components(self) -> None:
        """Инициализирует компоненты для серверного режима."""
        self.logger.info("Инициализация серверных компонентов...")

        try:
            # Инициализация Supabase
            self.supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

            # Инициализация Firebase
            self.cred = credentials.Certificate("./data/facematch-1a7a4-8af3d7baf28e.json")
            firebase_admin.initialize_app(self.cred)
            self.db = firestore.client()
            self.logger.info("Firebase и Supabase успешно инициализированы")
        except Exception as e:
            self.logger.error(f"Ошибка инициализации серверных компонентов: {str(e)}", exc_info=True)
            raise

    def _start_monitoring(self) -> None:
        """Запускает мониторинг новых файлов в отдельном потоке."""
        self.logger.info("Запуск мониторинга...")
        self.monitor_thread = threading.Thread(target=self._monitor_files, daemon=True)
        self.monitor_thread.start()

    def _monitor_files(self) -> None:
        """Мониторит новые файлы в Supabase Storage."""
        last_checked = datetime.now(timezone.utc) - timedelta(minutes=5)
        retry_count = 0
        max_retries = 5
        base_delay = 2

        while True:
            try:
                self._check_new_files(last_checked)
                last_checked = datetime.now(timezone.utc)
                retry_count = 0
            except httpx.ReadError as e:
                retry_count += 1
                if retry_count > max_retries:
                    self.logger.error(f"Максимальное количество попыток ({max_retries}) превышено")
                    self._handle_critical_failure()
                    break

                delay = min(base_delay * (2 ** (retry_count - 1)), 60)  # Экспоненциальная задержка с ограничением
                self.logger.warning(
                    f"Ошибка подключения (попытка {retry_count}/{max_retries}). Повтор через {delay} сек. Ошибка: {str(e)}"
                )
                time.sleep(delay)
                continue
            except Exception as e:
                self.logger.error(f"Неожиданная ошибка мониторинга: {str(e)}", exc_info=True)
                time.sleep(10)
                continue

            time.sleep(5)

    def _check_new_files(self, last_checked: datetime) -> None:
        """Проверяет наличие новых файлов."""
        self.logger.debug("Проверка новых файлов...")
        files = self.supabase.storage.from_('user-photo').list('uploads')
        new_files = [
            f for f in files
            if datetime.strptime(f['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc) > last_checked
        ]

        for file in new_files:
            self._process_storage_file(file)

    def _process_storage_file(self, file: Dict[str, Any]) -> None:
        """Обрабатывает файл из хранилища."""
        file_name = file['name']
        file_path = f"uploads/{file_name}"

        try:
            self.logger.info(f"Обработка файла: {file_path}")
            file_data = self.supabase.storage.from_('user-photo').download(file_path)
            self.process_image_file(file_data, file_name.split('.')[0])

            # Удаление после обработки
            self.supabase.storage.from_('user-photo').remove([file_path])
            self.logger.info(f"Файл {file_path} удален из хранилища")
        except Exception as e:
            self.logger.error(f"Ошибка обработки файла {file_path}: {str(e)}", exc_info=True)

    def open_image(self) -> None:
        """Открывает изображение через диалоговое окно."""
        if self.is_camera_active:
            self.stop_camera()

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            "JPEG Images (*.jpg *.jpeg);;All Files (*)"
        )

        if file_name:
            self.process_image_file(file_name)

    def process_image_file(self, data: Any, id_photo: Optional[str] = None) -> None:
        """Обрабатывает изображение из файла или данных."""
        try:
            # Преобразование в numpy array
            image = self._load_image_data(data)
            if image is None:
                return

            # Обработка изображения
            result = self._process_image(image)
            if result is None:
                return

            # Обработка результатов
            if self.is_server:
                self._handle_server_result(result, id_photo)
            else:
                self._handle_gui_result(result, image)

        except Exception as e:
            self.logger.error(f"Ошибка обработки изображения: {str(e)}", exc_info=True)

    def _load_image_data(self, data: Any) -> Optional[np.ndarray]:
        """Загружает изображение из файла или байтов."""
        try:
            if not self.is_server:
                with open(data, 'rb') as stream:
                    bytes_data = bytearray(stream.read())
            else:
                bytes_data = bytearray(data)

            array = np.asarray(bytes_data, dtype=np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)

            if image is None:
                self.logger.warning("Не удалось декодировать изображение")

            return image
        except Exception as e:
            self.logger.error(f"Ошибка загрузки изображения: {str(e)}", exc_info=True)
            return None

    def _process_image(self, image: np.ndarray) -> Optional[Tuple]:
        """Обрабатывает изображение через нормализатор."""
        try:
            result = self.normalizer.process_image(image)
            if result is None:
                return [], [], []
            else:
                return result
        except Exception as e:
            self.logger.error(f"Ошибка обработки изображения: {str(e)}", exc_info=True)
            return None

    def _handle_server_result(self, result: Tuple, id_photo: str) -> None:
        """Обрабатывает результаты для серверного режима с надежной транзакцией."""
        faces, norms, _ = result
        output = []

        data = {'result': []}
        for face, norm in zip(faces, norms):
            emb = self.normalizer.get_embedding(norm)
            best_labels, best_score = self._find_best_match(emb)
            data['result'].append({
                'pos': [face.left(), face.top(), face.right(), face.bottom()],
                'labels': best_labels,
                'score': best_score
            })

        if not data['result']:
            data['result'].append({
                'labels': {'Нет лица'}
            })

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            transaction = None
            try:
                doc_ref = self.db.collection("people").document(id_photo)
                transaction = self.db.transaction()

                @firestore.transactional
                def transactional_update(transaction: Transaction, doc_ref) -> None:
                    """Транзакционное обновление документа."""
                    snapshot = doc_ref.get(transaction=transaction)
                    if snapshot.exists:
                        existing_data = snapshot.to_dict()
                        # Сохраняем только если документ не изменялся
                        if all(k not in existing_data for k in data.keys()):
                            transaction.update(doc_ref, data)
                        else:
                            self.logger.warning(f"Документ {id_photo} уже содержит данные, пропускаем обновление")
                    else:
                        transaction.set(doc_ref, data)

                # Выполняем транзакцию
                transactional_update(transaction, doc_ref)
                self.logger.info(f"Данные успешно записаны (попытка {attempt + 1})")
                return

            except Exception as e:
                self.logger.error(f"Ошибка транзакции (попытка {attempt + 1}): {str(e)}")

            time.sleep(retry_delay)

        # Если все попытки исчерпаны
        self.logger.error(f"Не удалось сохранить данные для {id_photo} после {max_retries} попыток")

    def _handle_gui_result(self, result: Tuple, image: np.ndarray) -> None:
        """Обрабатывает результаты для GUI режима."""
        faces, norms, _ = result

        for face, norm in zip(faces, norms):
            emb = self.normalizer.get_embedding(norm)
            best_labels, best_score = self._find_best_match(emb)

            # Рисуем прямоугольник и текст
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{', '.join(best_labels)}: {best_score:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.display_image(image)

    def _find_best_match(self, embedding: np.ndarray) -> Tuple[List[str], float]:
        """Находит наиболее похожий эмбеддинг."""
        similarities = [1 - cosine(embedding, x) for x in self.embeddings]
        max_similarity = max(similarities)
        best_idx = np.arange(len(similarities))[similarities == max_similarity]
        return list(self.labels[best_idx]), float(max_similarity)

    def display_image(self, cv_image: np.ndarray) -> None:
        """Отображает изображение в GUI."""
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qt_image)
        self.resize_image()

    def resize_image(self) -> None:
        """Изменяет размер изображения для отображения."""
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)


if __name__ == "__main__":
    config.logger.getChild('main').info("Запуск приложения...")
    load_dotenv()
    IS_SERVER = 1 == 1

    try:
        app = QApplication(sys.argv)
        window = FaceRecognitionApp(IS_SERVER)
        if not IS_SERVER:
            window.show()
        sys.exit(app.exec_())
    except Exception as e:
        config.logger.critical(f"Критическая ошибка: {str(e)}", exc_info=True)
        sys.exit(1)
