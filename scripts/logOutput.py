import logging

class OutputLogger:
    def __init__(self, name="root", level="INFO", fname=None, format=None):
        self.logger = logging.getLogger(name)
        self.name = self.logger.name
        self.level = getattr(logging, level)
        self.logger.setLevel(self.level)

        # formatter = logging.Formatter(format)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        if formatter is not None:
            formatter = logging.Formatter(format)
        if fname is not None:
            fh = logging.FileHandler(fname)
            fh.setLevel(self.level)
            if format is not None:
                fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def write(self, msg):
        if msg and not msg.isspace():
            self.logger.log(self.level, msg)

    def flush(self): pass
