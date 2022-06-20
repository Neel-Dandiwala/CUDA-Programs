#include "thread_example.h"

void doWork(int threadIndex){
    std::cout << "Performing work for thread: " << threadIndex << std::endl;
}

void executeThreads(){
    std::cout << "Starting threads\n";

    std::thread zeroth(doWork, 0);
    std::thread first(doWork, 1);
    std::thread second(doWork, 2);

    std::cout << "Keyboard Interrupt to proceed the program\n";
    getchar();

    std::cout << "Joining threads\n";

    second.join();
    zeroth.join();
    first.join();

    std::cout << "Completed all threads\n";
}

void executeAndDetachThread() {
    std::thread thread(doWork, 0);
    thread.detach();
}

int main() {
    executeThreads();
    executeAndDetachThread();

    std::cout << "Sleeping for 1 second \n";
    std::this_thread::sleep_for (std::chrono::seconds(1));

    std::cout << "Press a key to proceed\n";
    getchar();
    return 0;
}