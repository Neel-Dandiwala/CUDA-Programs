import numpy as np 
import pandas as pd 
import multiprocessing as mp 
from pathos.multiprocessing import ProcessingPool as Pool

def prepare_data():
    print(f"Number of processes: {mp.cpu_count()}")
    r = np.random.RandomState(100)
    df = pd.DataFrame(r.randint(3, 10, size=[500,2]))
    print(df.head())
    return df
    
def hypotenuse(row):
    return round(row[1]**2 + row[2]**2, 2)**0.5

def sum_sqr(column):
    return sum([i**2 for i in column[1]])
    
def func(df):
    return df.shape

def main():
    result =[]
    output = []
    df = prepare_data()
    cores = mp.cpu_count()
    df_split = np.array_split(df, cores, axis=0) 
    
    # For row
    """
    with mp.Pool(4) as pool:
        result = pool.imap(hypotenuse, df.itertuples(index=True, name=None), chunksize=10)
        output = [round(x, 2) for x in result]
    """
    
    # For column
    """
    with mp.Pool(2) as pool:
        result = pool.imap(sum_sqr, df.iteritems(), chunksize=10)
        output = [x for x in result]
    """
    
    # For function
    pool = Pool(cores)
    df_out = np.vstack(pool.map(func, df_split))
    pool.close()
    pool.join()
    pool.clear()
    
    print(output)
    
if __name__ == '__main__':
    main()