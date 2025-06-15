import torch

from torch import nn
from typing import Optional
from config import config


class FaceAutoencoder(nn.Module):
    """Autoencoder для работы с лицами."""
    def __init__(self):
        super().__init__()
        self.logger = config.logger.getChild('FaceAutoencoder')

        self.logger.info(f"Инициализация FaceAutoencoder с latent_dim={config.LATENT_DIM}")

        # Инициализация encoder и decoder
        self.encoder = self._build_encoder(config.LATENT_DIM)
        self.decoder = self._build_decoder(config.LATENT_DIM)

        self.logger.debug("Модель успешно инициализирована")

    def _build_encoder(self, latent_dim: int) -> nn.Sequential:
        """Строит encoder часть autoencoder."""
        self.logger.debug("Построение encoder...")
        return nn.Sequential(
            self._conv_block(3, 32),    # [3, 256, 256]  -> [32, 128, 128]
            self._conv_block(32, 64),   # [32, 128, 128] -> [64, 64, 64]
            self._conv_block(64, 128),  # [64, 64, 64]   -> [128, 32, 32]
            self._conv_block(128, 256), # [128, 32, 32]  -> [256, 16, 16]
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, latent_dim)
        )

    def _build_decoder(self, latent_dim: int) -> nn.Sequential:
        """Строит decoder часть autoencoder."""
        self.logger.debug("Построение decoder...")
        return nn.Sequential(
            nn.Linear(latent_dim, 256 * 16 * 16),
            nn.Unflatten(1, (256, 16, 16)),
            self._deconv_block(256, 128),   # [256, 16, 16]  -> [128, 32, 32]
            self._deconv_block(128, 64),    # [128, 32, 32]  -> [64, 64, 64]
            self._deconv_block(64, 32),     # [64, 64, 64]   -> [32, 128, 128]
            self._deconv_block(32, 3,       # [32, 128, 128] -> [3, 256, 256]
                               activation=nn.Tanh(),
                               batch_norm=False)
        )

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Создает блок свертки для encoder."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _deconv_block(self,
                      in_channels: int,
                      out_channels: int,
                      activation: Optional[nn.Module] = nn.ReLU(inplace=True),
                      batch_norm: bool = True) -> nn.Sequential:
        """Создает блок транспонированной свертки для decoder."""
        layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            layers.append(activation)
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через autoencoder."""
        self.logger.debug(f"Входной тензор формы: {x.shape}")

        latent = self.encoder(x)
        self.logger.debug(f"Латентное представление формы: {latent.shape}")

        reconstructed = self.decoder(latent)
        self.logger.debug(f"Реконструированный тензор формы: {reconstructed.shape}")

        return reconstructed

    def get_config_summary(self) -> dict:
        """Возвращает конфигурацию модели в виде словаря."""
        _config = {
            'encoder_layers': len(list(self.encoder.children())),
            'decoder_layers': len(list(self.decoder.children())),
            'latent_dim': config.LATENT_DIM,
            'input_shape': (3, 256, 256)
        }
        self.logger.debug(f"Конфигурация модели: {_config}")
        return config


if __name__ == "__main__":
    model = FaceAutoencoder()
