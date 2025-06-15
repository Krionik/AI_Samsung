import cv2
import dlib
import numpy as np
import skimage.transform as tr
import torch

from torchvision import transforms
from config import config
from model import FaceAutoencoder
from typing import Optional, Tuple, List


class FaceNormalizer:
    """Класс для нормализации лиц и извлечения эмбеддингов."""

    def __init__(self):
        self.logger = config.logger.getChild('FaceNormalizer')

        # Инициализация модели
        self._init_model()

        # Инициализация детектора и предиктора
        self._init_detectors()

        # Загрузка шаблона лица
        self._load_face_template()

        self.logger.info("FaceNormalizer успешно инициализирован")

    def _init_model(self) -> None:
        """Инициализирует модель autoencoder."""
        try:
            self.logger.info("Загрузка модели FaceAutoencoder...")
            self.loaded_model = FaceAutoencoder()

            self.logger.debug(f"Путь к модели: {config.MODEL_PATH}")

            self.loaded_model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
            self.loaded_model.eval()
            self.loaded_model = self.loaded_model.to(config.DEVICE)

            self.logger.info("Модель успешно загружена")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {str(e)}", exc_info=True)
            raise

    def _init_detectors(self) -> None:
        """Инициализирует детекторы лиц и ключевых точек."""
        try:
            self.logger.debug("Инициализация детекторов...")
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(config.SHAPE_PREDICTOR_PATH)
        except Exception as e:
            self.logger.error(f"Ошибка инициализации детекторов: {str(e)}", exc_info=True)
            raise

    def _load_face_template(self) -> None:
        """Загружает шаблон лица и вычисляет референсные точки."""
        try:
            self.logger.debug("Загрузка шаблона лица...")
            self.face_template = np.load(config.FACE_TEMPLATE_PATH)
            self.ref_landmarks = self.face_template[[39, 42, 57]] * config.IMAGE_SIZE
        except Exception as e:
            self.logger.error(f"Ошибка загрузки шаблона: {str(e)}", exc_info=True)
            raise

    def _calculate_affine_transform(self, src_points: np.ndarray) -> Optional[np.ndarray]:
        """Вычисляет аффинное преобразование для нормализации."""
        try:
            A = np.hstack([src_points, np.ones((3, 1))]).astype(np.float64)
            B = np.hstack([self.ref_landmarks, np.ones((3, 1))]).astype(np.float64)
            transform = np.linalg.solve(A, B).T
            self.logger.debug("Аффинное преобразование вычислено")
            return transform
        except np.linalg.LinAlgError as e:
            self.logger.warning(f"Ошибка вычисления преобразования: {str(e)}")
            return None

    def normalize(self,
                  img: np.ndarray,
                  only_one: bool = False
                  ) -> Optional[Tuple[List[dlib.rectangle], List[np.ndarray]]]:
        """Нормализует лица на изображении."""
        try:
            self.logger.debug("Начало нормализации изображения...")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            faces = self.detector(gray)
            if not faces:
                self.logger.warning("Лица не обнаружены")
                return None

            if only_one:
                faces = faces[:1]
                self.logger.debug("Обработка только одного лица")

            result = []
            for i, face in enumerate(faces):
                points = self.predictor(gray, face)
                landmarks = np.array([[p.x, p.y] for p in points.parts()])
                current_landmarks = landmarks[[39, 42, 57]]

                T = self._calculate_affine_transform(current_landmarks)
                if T is None:
                    return None

                img_normalized = img_rgb.astype(np.float32) / 255.0
                wrapped = tr.warp(
                    img_normalized,
                    tr.AffineTransform(T).inverse,
                    output_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                    order=3,
                    mode='constant',
                    cval=0,
                    preserve_range=True
                )

                result_img = (wrapped * 255).astype(np.uint8)
                result.append(result_img)
                self.logger.debug(f"Лицо {i + 1} успешно нормализовано")

            return faces, result

        except Exception as e:
            self.logger.error(f"Ошибка нормализации: {str(e)}", exc_info=True)
            return None

    def process_image(self,
                      image: np.ndarray
                      ) -> Optional[Tuple[List[dlib.rectangle], List[np.ndarray], List[np.ndarray]]]:
        """Обрабатывает изображение: нормализация и реконструкция."""
        try:
            self.logger.debug("Начало обработки изображения...")
            norm_result = self.normalize(image)
            if norm_result is None:
                return None

            faces, norms = norm_result
            outputs = []

            for i, norm in enumerate(norms):
                self.logger.debug(f"Обработка нормализованного лица {i + 1}")
                tensor = transforms.ToTensor()(norm).unsqueeze(0).to(config.DEVICE)

                with torch.no_grad():
                    reconstructed = self.loaded_model(tensor)

                output = (reconstructed.squeeze().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
                outputs.append(output.astype(np.uint8))
                self.logger.debug(f"Лицо {i + 1} успешно реконструировано")

            self.logger.debug("Обработка изображения завершена")
            return faces, norms, outputs

        except Exception as e:
            self.logger.error(f"Ошибка обработки изображения: {str(e)}", exc_info=True)
            return None

    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Извлекает эмбеддинг лица."""
        try:
            self.logger.debug("Извлечение эмбеддинга...")
            tensor = transforms.ToTensor()(image).unsqueeze(0).to(config.DEVICE)

            with torch.no_grad():
                emb = self.loaded_model.encoder(tensor)

            embedding = emb.cpu().numpy()[0]
            self.logger.debug("Эмбеддинг успешно извлечен")
            return embedding

        except Exception as e:
            self.logger.error(f"Ошибка извлечения эмбеддинга: {str(e)}", exc_info=True)
            return None


if __name__ == "__main__":
    normalizer = FaceNormalizer()
