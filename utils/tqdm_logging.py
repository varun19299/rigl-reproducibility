from tqdm import tqdm


class TqdmStream(object):
    @classmethod
    def write(_, msg):
        tqdm.write(msg, end="")
