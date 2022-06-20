#include "future_example.h"

void doWorkWithFutures(std::future<int>& fut){
    int x = fut.get();
    std::cout << "Future index: " << x << " executing\n";
}

void executeThreadsWithFutures(){
    std::cout << "Starting threads to use futures\n";

    std::promise<int> prom; // create promise 
    std::future<int> fut = prom.get_future();

    std::thread th1(doWorkWithFutures, std::ref(fut));  

    std::cout << "Keyboard interrupt to proceed the program\n";
    getchar();

    prom.set_value(1);
    th1.join();

    std::cout << "Keyboard interrupt to proceed the program\n";
    getchar();
    std::cout << "Completed all threads\n";
}

int doWorkWithAsync(int x){
    std::cout << "Squaring " << x << ", Please wait...\n";
    return x*x;
}

void executeWithAsync(int numFutures) {
    std::cout << "Using future with async to asynchronously execute squaring of values" << "\n";

    std::future<int> first = std::async (doWorkWithAsync, 1);
    std::future<int> second = std::async (doWorkWithAsync, 2);
    std::future<int> third = std::async (doWorkWithAsync, 3);

    int firstResult = first.get();
    std::cout << "Future first squared value: " << firstResult << std::endl;
    int secondResult = second.get();
    std::cout << "Future second squared value: " << secondResult << std::endl;
    int thirdResult = third.get();
    std::cout << "Future third squared value: " << thirdResult << std::endl;
    
}

int main(int argc, char *argv[]){
    int numFutures = 3;
    if(argc > 1){
        numFutures = atoi(argv[1]);
    }
    
    executeThreadsWithFutures();

    std::cout << "Press a key to proceed\n";
    getchar();

    executeWithAsync(numFutures);

    std::cout << "Press a key to proceed\n";
    getchar();

    return 0;
}