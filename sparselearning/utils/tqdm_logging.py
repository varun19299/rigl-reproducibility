import logging

from tqdm import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    """
    Handler to pass tqdm outputs to the output file
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
