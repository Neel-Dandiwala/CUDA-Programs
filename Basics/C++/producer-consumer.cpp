#include <iostream>
#include <thread>
#include <deque>
#include <mutex>
#include <chrono> // Date, timers and clocks
#include <condition_variable> // Synchronisation primitive

using std::deque; 
std::mutex mu, cout_mu;
std::condition_variable cond;

class Buffer
{
    public:
        void add(int num) { 
            while(true) {
                std::unique_lock<std::mutex> locker(mu);
                cond.wait(locker, [this](){return buffer_.size() < size_;});
                buffer_.push_back(num);
                locker.unlock();
                cond.notify_all();
                return;
            }
        }

        int remove(){
            while(true) {
                std::unique_lock<std::mutex> locker(mu);
                cond.wait(locker, [this](){return buffer_.size() > 0;});
                int back = buffer_.back();
                buffer_.pop_back();
                locker.unlock();
                cond.notify_all();
                return back;
            }
        }

        Buffer() {}

    private:
        deque<int> buffer_;
        const unsigned int size_ = 10;
};

class Producer
{
    public:
        Producer(Buffer* buffer, std::string name){
            this->buffer_ = buffer;
            this->name_ = name;
        }
        void run() {
            while(true) {
                int num = std::rand() % 100;
                buffer_->add(num);
                cout_mu.lock();
                int sleep_time = std::rand() % 100;
                std::cout << "Name: " << name_ << "\t Produced: " << num << "\t Sleep Time: " << sleep_time << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
                cout_mu.unlock();
            }
        }
    
    private:
        Buffer* buffer_;
        std::string name_;
};

class Consumer
{
    public: 
        Consumer(Buffer* buffer, std::string name) {
            this->buffer_ = buffer;
            this->name_ = name;
        }
        void run() {
            while(true) {
                int num = buffer_->remove();
                cout_mu.lock();
                int sleep_time = std::rand() % 100;
                std::cout << "Name: " << name_ << "\t Consumed: " << num << "\t Sleep Time: " << sleep_time << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
                cout_mu.unlock();
            }
        }

    private:
        Buffer* buffer_;
        std::string name_;
};

int main() {
    
    Buffer b;
    Producer p1(&b, "Producer1");
    Producer p2(&b, "Producer2");
    Producer p3(&b, "Producer3");
    Consumer c1(&b, "Consumer1");
    Consumer c2(&b, "Consumer2");
    Consumer c3(&b, "Consumer3");

    std::thread producer_thread1(&Producer::run, &p1); //Function and argument
    std::thread producer_thread2(&Producer::run, &p2);
    std::thread producer_thread3(&Producer::run, &p3);

    std::thread consumer_thread1(&Consumer::run, &c1);
    std::thread consumer_thread2(&Consumer::run, &c2);
    std::thread consumer_thread3(&Consumer::run, &c3);

    //Wait for the thread to finish, this is a blocking operation
    producer_thread1.join();
    producer_thread2.join();
    producer_thread3.join();
    consumer_thread1.join();
    consumer_thread2.join();
    consumer_thread3.join();

    return 0;
}