import logging
import matplotlib.pyplot as plt
import numpy as np
import contextlib
import sys
from tqdm import tqdm
from time import sleep
from datetime import datetime

from logOutput import OutputLogger

def main_old():
    print('Outside of context')
    format = '%(levelname)s:%(message)s'
    now = datetime.now().strftime('%d%m%Y-%H%M%S')

    logging.basicConfig(level=logging.DEBUG, filename=f'{now}-test.txt', format=format)
    for i in tqdm(range(3)):
        with contextlib.redirect_stdout(OutputLogger('my_logger', 'DEBUG')):
            print(f'{i}s elapsed')
        sleep(1)

    print('Back outside of context')
    x = np.linspace(0, 2*np.pi, 100)
    plt.plot(x, np.sin(x))
    plt.show()

if __name__=="__main__":
    format = '%(levelname)s:%(message)s'
    now = datetime.now().strftime('%d%m%Y-%H%M%S')
    fname = f'{now}-test.txt'
    print('About to log')
    for i in tqdm(range(3)):
        with contextlib.redirect_stdout(OutputLogger('my_logger', 'DEBUG', fname=fname, format=format)):
            print(f'{i}s elapsed')
        sleep(1)

    print('Finished logging')
    print('Writing test logger')
    logger = OutputLogger('logger', fname='test.txt', format=format, level='DEBUG')
    logger.write('Test message')
