import logging
from tqdm import tqdm

class TqdmLoggingHandler(logging.StreamHandler):
    # def __init__(self, level=logging.NOTSET):
    #     super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
