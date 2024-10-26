#include <iostream>
#include <iomanip>  // for std::hex
# include <cstdint>

void print_uint128(__uint128_t value) {
    if (value == 0) {
        std::cout << "0";
        return;
    }
    
    // Split the 128-bit value into two 64-bit parts
    uint64_t high = value >> 64;  // The higher 64 bits
    uint64_t low = value & 0xFFFFFFFFFFFFFFFFULL;  // The lower 64 bits

    if (high != 0) {
        std::cout << std::hex << high << std::setw(16) << std::setfill('0') << low << std::dec;  // Print both high and low in hex
    } else {
        std::cout << low;  // If high is 0, print just the low part
    }
}

int main() {
    __uint128_t result = static_cast<__uint128_t>(1) << 127;  // This is 2^128

    std::cout << "2^128 is: ";
    print_uint128(result);  // Call our function to print the 128-bit value
    std::cout << std::endl;

    return 0;
}