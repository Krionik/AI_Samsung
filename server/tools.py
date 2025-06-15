import cv2
import numpy as np

from norm import FaceNormalizer
from config import config
from pathlib import Path
from typing import List, Tuple, Optional


def load_embeddings(base_dir: str) -> Tuple[List[str], List[np.ndarray]]:
    """
    Загружает изображения из директории, нормализует их и извлекает эмбеддинги.
    """
    normalizer = FaceNormalizer()
    logger.info("Инициализирован нормализатор лиц")

    labels: List[str] = []
    embeddings: List[np.ndarray] = []
    processed_count = 0
    error_count = 0

    base_path = Path(base_dir)
    if not base_path.is_dir():
        logger.error(f"Директория не найдена: {base_dir}")
        raise FileNotFoundError(f"Директория {base_dir} не существует")

    logger.info(f"Начата обработка изображений из {base_dir}")

    for label_dir in base_path.iterdir():
        if not label_dir.is_dir():
            continue

        logger.debug(f"Обработка директории: {label_dir.name}")

        for img_path in label_dir.iterdir():
            if not img_path.is_file():
                continue

            # Загрузка изображения
            img_array = load_image(img_path)
            if img_array is None:
                error_count += 1
                continue

            # Обработка и извлечение эмбеддинга
            embedding = process_image(normalizer, img_array)
            if embedding is not None:
                labels.append(label_dir.name)
                embeddings.append(embedding)
                processed_count += 1
            else:
                error_count += 1

    logger.info(f"Обработка завершена. Успешно: {processed_count}, Ошибок: {error_count}")
    return labels, embeddings


def load_image(img_path: Path) -> Optional[np.ndarray]:
    """Загружает изображение с обработкой ошибок."""
    try:
        img_array = np.fromfile(img_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        if image is None:
            logger.warning(f"Не удалось декодировать изображение: {img_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Ошибка загрузки {img_path}: {str(e)}", exc_info=True)
        return None


def process_image(normalizer: FaceNormalizer, image: np.ndarray) -> Optional[np.ndarray]:
    """Обрабатывает изображение и извлекает эмбеддинг."""
    try:
        result = normalizer.process_image(image)
        if result is None:
            return None

        _, norm, _ = result
        embedding = normalizer.get_embedding(norm[0])
        return embedding
    except Exception as e:
        logger.error(f"Ошибка обработки изображения: {str(e)}", exc_info=True)
        return None


def save_embeddings(labels: List[str],
                    embeddings: List[np.ndarray],
                    output_path: str = "./data/embeddings.npz"
                    ) -> None:
    """Сохраняет эмбеддинги в сжатый npz-файл."""
    try:
        np.savez_compressed(
            output_path,
            labels=np.array(labels),
            embeddings=np.array(embeddings)
        )
        logger.info(f"Эмбеддинги успешно сохранены в {output_path}")
    except Exception as e:
        logger.error(f"Ошибка сохранения эмбеддингов: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    logger = config.logger.getChild('tools')
    try:
        logger.info("Запуск обработки изображений")
        labels, embeddings = load_embeddings("<путь к директории с изображениями>")
        save_embeddings(labels, embeddings)
        logger.info("Программа завершена успешно")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)
        raise
