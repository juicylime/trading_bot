import logging

class Logger:
    def __init__(self, name, file):
        self.logger = logging.getLogger(name)
        if not self.logger.hasHandlers():
            handler = logging.FileHandler(file)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def get_logger(self):
        return self.logger