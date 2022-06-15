import threading
import random
import time

class Philosopher(threading.Thread):
    running = True 
    
    def __init__(self, xname, chopstick_left, chopstick_right):
        threading.Thread.__init__(self)
        self.name = xname
        self.chopstick_left = chopstick_left
        self.chopstick_right = chopstick_right
        
    def run(self):
        while self.running:
            time.sleep(random.uniform(3, 13))
            print(f'{self.name} is hungry.')
            self.dine()
            
    def dine(self):
        chopstick1, chopstick2 = self.chopstick_left, self.chopstick_right
        while self.running:
            chopstick1.acquire(True)
            locked = chopstick2.acquire(False)
            if locked:
                break
            chopstick1.release()
            print(f'{self.name} swaps chopsticks')
            chopstick1, chopstick2 = chopstick2, chopstick1
        else:
            return
        
        self.dining()
        chopstick2.release()
        chopstick1.release()
        
    def dining(self):
        print(f'{self.name} starts eating')
        time.sleep(random.uniform(1,10))
        print(f'{self.name} finished eating')
        
def dining_philosophers():
    
    chopsticks = [threading.Lock() for n in range(5)]
    philosophers = ('Jiang', 'Nishida', 'Zhu', 'Egonu', 'Li')
    
    philosophers = [Philosopher(philosophers[i], chopsticks[i % 5], chopsticks[(i + 1) % 5]) for i  in range(5)]
    
    random.seed(507129)
    Philosopher.running = True
    for p in philosophers:
        p.start()
    time.sleep(100)
    Philosopher.running = False
    print("Dinner is over")

dining_philosophers()