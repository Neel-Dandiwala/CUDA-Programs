#include "atomic_example.h"

void doWorkWithAtomicBoolean(int threadIndex){
    std::cout << "Starting work with thread: " << threadIndex << " will wait for atomic boolean\n";
    while (!ready) {
        std::this_thread::yield(); //wait for ready signal
    }
    ready = false;
    std::cout << "Completed work for thread: " << threadIndex << std::endl;
}

void executeThreadsWithAtomicBoolean(int numThreads) {
    std::cout << "Starting " << numThreads << " threads to use shared atomic boolean\n";
    for (int index = 0; index < numThreads; ++index) {
        std::thread(doWorkWithAtomicBoolean, index).detach();
    }
    std::cout << "Keyboard Interrupt to proceed the program\n";
    getchar();

    for (int i = 0; i < numThreads; ++i){
        ready = true;
        std::cout << "Sleeping for 1 second\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "Keyboard Interrupt to proceed the program\n";
    getchar();
    std::cout << "Completed all threads \n";
}

void doWorkWithAtomicThreadFence(int threadIndex) {
    std::cout << "Starting work for thread: " << threadIndex << " will wait for atomic_thread_fence to be relaxed\n";
    while (!ready.load(std::memory_order_relaxed)){
        std::this_thread::yield();
    }
    atomic_thread_fence(std::memory_order_acquire);
    std::cout << "Completed work for thread: " << threadIndex << " after atomic_thread_fence was acquired\n";
}

void executeWithAtomicThreadFence(int numThreads) {
    std::cout << "Starting " << numThreads << " threads to use shared atomic_thread_fence\n";
    for (int i = 0; i < numThreads; ++i) {
        std::thread(doWorkWithAtomicThreadFence, i).detach();
    }
    atomic_thread_fence(std::memory_order_release);
    std::cout << "Keyboard interrupt to proceed the program\n";
    getchar();

    std::cout << "Setting boolean variable ready to true to allow each thread to continue\n";
    for (int i = 0; i < numThreads; ++i){
        ready = true;
        std::cout << "Sleeping for 1 second\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "Keyboard interrupt to proceed the program\n";
    getchar();
    std::cout << "Completed all threads\n";
}

int main(int argc, char* argv[]) {
    int numThreads = 3;
    if (argc > 1){
        numThreads = atoi(argv[1]);
    }
    executeThreadsWithAtomicBoolean(numThreads);
    std::cout << "\n\n\n\n";
    executeWithAtomicThreadFence(numThreads);

    std::cout << "Sleeping for 1 second\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "Press a key to let program proceed\n";
    getchar();
    return 0;
}