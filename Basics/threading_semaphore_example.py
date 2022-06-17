import logging 
import time
import random
import sys    
from threading import BoundedSemaphore, Thread, active_count
import pydash as _

def thread_function(name):
    logging.info(f"Thread starting: {name}")
    time.sleep(1)
    logging.info(f"Thread finished: {name}")

def critical_section_acquire_release(name, sync_object):
     time.sleep(random.randint(0,10))
     sync_object.acquire()
     logging.info(f"Critical Section: Thread {name} acquired the semaphore")
     thread_function(name)
     sync_object.release()
     logging.info(f"Critical Section: Thread {name} released the semaphore")
     
class ThreadingSemaphoreExample:
    def __init__(self, num_thread, semaphore_size):
        self.num_thread = num_thread
        self.semaphore_size = semaphore_size
        self.semaphore = BoundedSemaphore(value=self.semaphore_size)
        _format = "%(asctime)s : %(message)s"
        logging.basicConfig(format=_format, level=logging.INFO, datefmt="%H:%M:%S")
        
    def run(self):
        threads = list()
        initial_num_threads = active_count()
        for index in range(self.num_thread):     
            logging.info(f"Thread run: Create and Start Thread {index}")
            thread = Thread(group=None, target=critical_section_acquire_release, args=(index, self.semaphore))   
            threads.append(thread)
            thread.start()
            
        while active_count() > initial_num_threads:
            logging.info(f"Waiting for active threads to be done. Still left: {active_count() - initial_num_threads}")
            time.sleep(1)
            
        logging.info(f"There are no longer any active threads, program shall exit")
        
def main():
    thread_semaphore = ThreadingSemaphoreExample(3, 1)
    thread_semaphore.run()
    
if __name__ == '__main__':
    main()