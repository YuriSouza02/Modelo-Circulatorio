import logging


def setup_logging(level=logging.INFO, **kwargs):
    # Configura o sistema de logs padrão do Python definindo o formato da mensagem.
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)