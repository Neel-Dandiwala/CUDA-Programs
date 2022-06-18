import multiprocessing as mp 
import numpy as np

def common_items(row1, row2):
    return list(set(row1).intersection(row2))

def normalise(row):
    minimum = min(row)
    maximum = max(row)
    return [(num - minimum) / (maximum - minimum) for num in row]

def main():
    list_a = [[1, 2, 3], [5, 6, 7, 8], [10, 11, 12], [20, 21]]
    list_b = [[2, 3, 4, 5], [6, 9, 10], [11, 12, 13, 14], [21, 24, 25]]
    result = []
    pool = mp.Pool(mp.cpu_count())
    
    # Example 1 - Find Common Items
    """
    for row1, row2 in zip(list_a, list_b):
        result.append(pool.apply(common_items, args=(row1, row2)))
    """
    
    # Example 2 - Normalise the rows to a unit numbers
    result = [pool.apply(normalise, args=(row,)) for row in list_b]
    
    pool.close()
    print(result)
    
if __name__ == '__main__':
    main()