#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 1000000

long get_time_ms(void) {
    long ms;   // Milliseconds
    time_t s;  // Seconds
    struct timespec spec;

    clock_gettime(CLOCK_MONOTONIC, &spec);

    s = spec.tv_sec;
    ms = round(spec.tv_nsec / 1.0e6);  // Convert nanoseconds to milliseconds
    if (ms > 999) {
        s++;
        ms = 0;
    }

    return s * 1000L + ms;
}

int main(int argc, char* argv[]) {
    const int M = N % 2 == 0 ? (N - 4) / 2 + 1 : (N - 3) / 2 + 1;
    int checkStart = 3;
    // Check additional odd number to avoid inter-process communication
    int checkEnd = 3 + 2 * (M - 1);
    int checkSize = M;

    char* marked = (char*)malloc(checkSize);
    if (marked == NULL) {
        printf("Cannot allocate enough memory\n");
        return 1;
    }

    memset(marked, 0, checkSize);
    long startTime = 0, endTime = 0;
    startTime = get_time_ms();
    int prime = 3;
    int searchIndex = 0;  // search unmarked number from this index
    // Mark primes
    do {
        // Find the index of smallest multiple of prime
        int firstIdx;
        if (prime * prime > checkStart) {
            // index of prime*prime
            firstIdx = (prime * prime - checkStart) / 2;
        } else {
            if (checkStart % prime == 0)
                // index of checkStart
                firstIdx = 0;
            else {
                // index of first = `checkStart + (prime - (checkStart %
                // prime))` checkStart is always odd, if `first` is even,
                // then (prime - (checkStart % prime)) is odd
                int firstMultiple = (prime - (checkStart % prime));
                firstIdx = (firstMultiple % 2 == 0)
                               ? firstMultiple / 2
                               : (prime + firstMultiple) / 2;
            }
        }

        // Mark all multiple of prime
        for (int i = firstIdx; i < checkSize; i += prime) {
            marked[i] = 1;
        }
        // finds the next prime (smallest number that is not
        while (marked[++searchIndex])
            ;
        // index 0 is 3 (search range is [3, N])
        prime = 2 * searchIndex + 3;
    } while (prime * prime <= N);

    // Count primes
    int count = 0;
    for (int i = 0; i + 1 < checkSize; ++i) {
        if (!marked[i] && !marked[i + 1])
            // both 3+2*i and 3+2*(i+1) are prime
            ++count;
    }

    endTime = get_time_ms();

    printf("Number of consecutive odd prime pairs less than %d: %d\n", N,
           count);

    double elapsed_time = (endTime - startTime) / 1000.0;
    printf("Time elapsed: %f\n", elapsed_time);
    fflush(stdout);
    free(marked);
    return 0;
}