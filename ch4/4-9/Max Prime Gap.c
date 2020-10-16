#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 1000000

static bool isPrime(int n);

// Usage:
// mpicc -o [EXECUTABLE_NAME] __FILE__
// mpiexec -n [NUMBER_OF_PROCESS] [EXECUTABLE_NAME]

int main(int argc, char* argv[]) {
    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    unsigned char* primeFlags = (unsigned char*)calloc(N, sizeof(unsigned char));
    unsigned char* accPrimeFlags = (unsigned char*)calloc(N, sizeof(unsigned char));
    memset(primeFlags, 0, N * sizeof(unsigned char));
    memset(accPrimeFlags, 0, N * sizeof(unsigned char));

    for (int i = rank; i < N; i += np)
        primeFlags[i] = isPrime(i);

    MPI_Allreduce(primeFlags, accPrimeFlags, N, MPI_UNSIGNED_CHAR, MPI_LOR, MPI_COMM_WORLD);

    int maxGap = 0, accMaxGap = 0;
    int prevPrime = 0;
    int start = rank * N / np;
    int end = (rank + 1) * N / np;
    // each process calculates the gap of primes in [start, end) and its next prime (in range)
    // find the first prime
    for (int i = start; i < end; ++i) {
        if (accPrimeFlags[i]) {
            prevPrime = i;
            break;
        }
    }
    for (int i = prevPrime + 1; i < end; ++i) {
        if (accPrimeFlags[i]) {
            int gap = i - prevPrime;
            if (gap > maxGap)
                maxGap = gap;
            prevPrime = i;
        }
    }
    MPI_Reduce(&maxGap, &accMaxGap, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("The largest gap between two consecutive primes < 1,000,000 is: %d\n", accMaxGap);
        fflush(stdout);
    }

    free(primeFlags);
    free(accPrimeFlags);
    MPI_Finalize();
    return 0;
}

static bool isPrime(int n) {
    if (n == 1)
        return false;
    if (n % 2 == 0)
        return n == 2;
    for (int p = 3; p * p <= n; p += 2) {
        if (n % p == 0)
            return false;
    }
    return true;
}
