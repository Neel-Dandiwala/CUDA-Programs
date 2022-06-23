// Inverse Square Root of Number - Quake III Arena
#include <iostream>

float Qsqrt(float number){
    const float x2 = number * 0.5F;
    const float threehalfs = 1.5F;
  
    union {
        float f;
        uint32_t i;
    } conv = {number};
    
    conv.i = 0x5f3759df - (conv.i >> 1);
    conv.f *= (threehalfs - (x2 * conv.f * conv.f));
    
    return conv.f;
}

int main() {
    
    float y = Qsqrt(2.0F);
    std::cout << "Inverse Square Root = " << y << "\n";
    return 0;
}