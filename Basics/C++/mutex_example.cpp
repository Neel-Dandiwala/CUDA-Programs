#include "mutex_example.h"

void doWorkWithMutexLock(int threadIndex){
    sharedMutex.lock();
    std::cout << "Performing work for thread: " << threadIndex << std::endl;
    std::this_thread::sleep_for (std::chrono::seconds(1));
    sharedMutex.unlock();
}

void executeThreadsWithMutexLock(int numThreads) {
    std::thread threads[numThreads];
    std::cout << "Starting " << numThreads << " threads to use lock with shared mutex" << std::endl;
    int counter = 0;

    for (int i = 0; i < numThreads; ++i) {
        threads[i] = std::thread(doWorkWithMutexLock, i);
    }

    std::cout << "Keyboard Interrupt to proceed the program\n";
    getchar();

    std::cout << "Joining threads\n";
    for (int i = 0; i < numThreads; ++i){
        threads[i].join();
    }

    std::cout << "Keyboard interrupt to proceed the program\n";
    getchar();
    std::cout << "Completed all threads\n";
}

void doWorkWithMutexTryLock(int threadIndex) {
    int counter = 0;
    while(!sharedMutex.try_lock()){
        counter++;
        std::this_thread::sleep_for (std::chrono::seconds(1));
    }

    std::cout << "Counter for work for threads: " << threadIndex << " is " << counter << std::endl;
    sharedMutex.unlock();
}

void executeAndDetachThreadsWithMutexTryLock(int numThreads) {
    std::thread threads[numThreads];
    std::cout << "Starting " << numThreads << " threads with try_lock on shared mutex\n";
    int counter = 0;

    for (int i = 0; i < numThreads; ++i){
        threads[i] = std::thread(doWorkWithMutexTryLock, i);
    }
    std::cout << "Keyboard Interrupt to proceed the program\n";
    getchar();

    std::cout << "Detaching threads with try_lock on shared mutex\n";
    for (int i = 0; i < numThreads; ++i){
        threads[i].detach();
    }

    std::cout << "Keyboard Interrupt to proceed the program\n";
    getchar();
}

int main(int argc, char *argv[]) {
    int numThreads = 3;
    if(argc > 1){
        numThreads = atoi(argv[1]);
    }

    executeThreadsWithMutexLock(numThreads);

    executeAndDetachThreadsWithMutexTryLock(numThreads);

    std::cout << "Sleeping for 1 second\n";
    std::this_thread::sleep_for (std::chrono::seconds(1));

    std::cout << "Press a key to proceed\n";
    getchar();
    return 0;
}