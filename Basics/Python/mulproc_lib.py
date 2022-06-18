import multiprocessing as mp 
import time 
import numpy as np
import pandas as pd

# Prepare Data 

def prepare_data():
    print(f"Number of processes: {mp.cpu_count()}")
    r = np.random.RandomState(100)
    arr = r.randint(0, 10, size=[200000, 5])
    data = arr.tolist()
    print(data[:15])
    return data

# Without Parallelization

def count_within_range(row, minimum=3, maximum=7, i = -1):
    """ Returns how many numbers lie withing the limits in a given row """
    count = 0
    for n in row:
        if minimum <= n and n <= maximum:
            count = count + 1
    
    if i == -1:
        return count
    else:
        return (i, count)

def main():
    data = prepare_data()
    result = []

    print(f"Started at {time.strftime('%X')}")

    # Simple Way - No parallelization
    """
    for row in data:
        result.append(count_within_range(row, minimum = 3, maximum = 7))
    """ 

    # Synchronous Parallelization using Pool.apply()
    """
    pool = mp.Pool(mp.cpu_count())
    for row in data:
        result.append(pool.apply(count_within_range, args=(row, 4, 8)))

    pool.close()
    """ 
    
    # Synchronous Parallelization using Pool.map()
    """
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(count_within_range, [row for row in data])
    pool.close()
    """
    
    # Synchronous Parallelization using Pool.starmap()
    """
    # also accepts only one iterable as argument, but each element in that iterable is also a iterable
    pool = mp.Pool(mp.cpu_count())
    result = pool.starmap(count_within_range, [(row, 2, 6) for row in data])
    pool.close()
    """
    
    # Asynchronous Parallelization using Pool.apply_async()
    """
    pool = mp.Pool(mp.cpu_count())
    
    # With Callback
    for i, row in enumerate(data):
        pool.apply_async(count_within_range, args=(row, 3, 7, i), callback = lambda count: result.append(count))
    
    # Without Callback
    result_object = [pool.apply_async(count_within_range, args=(row, 3, 7, i)) for i, row in enumerate(data)]
    result = [r.get()[1] for r in result_object]
    
    pool.close()
    pool.join() # Postpones the execution of next line of code until all processes in the queue are done
    """
    
    # Asynchronous Parallelization using Pool.starmap_async()
    
    pool = mp.Pool(mp.cpu_count())
    result = pool.starmap_async(count_within_range, [(row, 2, 6, i) for i, row in enumerate(data)]).get()
    pool.close()
    
    print(f"Finished at {time.strftime('%X')}")
    print(result[:10])
    
if __name__ == '__main__':
    main()