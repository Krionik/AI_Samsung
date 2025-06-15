import torch
import logging
import colorlog
import signal
import sys


class Config:
    """Конфигурационный класс для параметров проекта."""

    # Инициализация системы логирования
    _log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    _formatter = colorlog.ColoredFormatter(
        '%(log_color)s' + _log_format,
        log_colors={
            'DEBUG': 'white',
            'INFO': 'blue',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

    # Настройка обработчиков
    _file_handler = logging.FileHandler('app.log')
    _console_handler = logging.StreamHandler()
    _file_handler.setFormatter(logging.Formatter(_log_format))
    _console_handler.setFormatter(_formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[_file_handler, _console_handler]
    )
    logger = logging.getLogger(__name__)

    SHAPE_PREDICTOR_PATH = "./data/shape_predictor_68_face_landmarks.dat"
    FACE_TEMPLATE_PATH = "./data/face_template.npy"

    # --- Параметры модели ---
    MODEL_PATH = "./data/model.pth"
    IMAGE_SIZE = 256
    LATENT_DIM = 1024

    def __init__(self):
        """Инициализация обработчиков сигналов."""
        signal.signal(signal.SIGTERM, self._handle_exit)
        signal.signal(signal.SIGINT, self._handle_exit)

    def _handle_exit(self, signum, frame):
        """Обработчик завершения работы."""
        self.logger.info(f"Получен сигнал {signum}, завершение работы...")
        sys.exit(0)

    # --- Вычисляемые свойства ---
    @property
    def DEVICE(self) -> torch.device:
        """Определяет доступное вычислительное устройство."""
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Методы ---
    @classmethod
    def print_config(self) -> None:
        """Выводит текущую конфигурацию."""
        width, left = 50, 15
        print(f"{' Configuration ':-^{width}}")
        print(f"{'Device:':<{left}} {self().DEVICE}")
        print(f"{'Model path:':<{left}} {self.MODEL_PATH}")
        print(f"{'Latent dim:':<{left}} {self.LATENT_DIM}")
        print(f"{'Image size:':<{left}} {self.IMAGE_SIZE}")
        print("-" * width)


config = Config()

if __name__ == "__main__":
    config.print_config()
