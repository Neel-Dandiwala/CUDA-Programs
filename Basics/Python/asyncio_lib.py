import asyncio
import time

async def async_func1():
    return 42

async def async_func2():
    return 6*7

async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)
    
async def main():
    print(f"Started at {time.strftime('%X')}")
    await say_after(1,'Hello World')
    await say_after(2,'I love donuts')
    print(f"Finished at {time.strftime('%X')}")
    
    task = asyncio.create_task(async_func1()) #coroutine in the background to run
    await task
    await asyncio.gather(async_func1(), async_func2())
    await asyncio.sleep(1)
    
    print(f"Completed at {time.strftime('%X')}")

asyncio.run(main())