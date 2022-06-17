import logging
import _thread 
import sys  
import time    

import pydash as _

def thread_function(index):
    logging.info(f"Thread Starting: {index}")
    time.sleep(1)
    logging.info(f"Thread Finishing: {index}")
    
class StartNewThreadExample:
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        _format = "%(asctime)s: %(message)s"
        logging.basicConfig(format=_format, level=logging.INFO, datefmt="%H:%M:%S")
        
    def run(self):
        sleeping_time = 0
        for index in range(self.num_threads):
            logging.info(f"Thread run: Create and Start Thread {index}")
            _thread.start_new_thread(thread_function, (index, ))
            sleeping_time += 1
        time.sleep(sleeping_time)
        logging.info(f"Threads completed")
 
def main():
    start_new_thread_example = StartNewThreadExample(5)
    start_new_thread_example.run()
           
if __name__ == "__main__":
    main()
    