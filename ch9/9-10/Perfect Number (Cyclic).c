#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// A perfect number is a positive integer whose value is equal to the sum of all
// its positive factors, excluding itself. The first two perfect numbers are 6
// and 28;

// The Greek mathematician Euclid (c. 300 BCE) showed that if  2^n — 1
// is prime, then (2^n - 1)2^(n-1)' is a perfect number. For example, 2^2 - 1 =
// 3 is prime, so (2^2 — 1)2^1' = 6 is a perfect number.

// Write a parallel program to find the first eight perfect numbers.

// how many perfect numbers to find
#define N 8
#define MAX_POW 50

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsedTime = -MPI_Wtime();

    for (int pow = rank; pow < MAX_POW; pow += size) {
        unsigned long n = (1ul << pow) - 1;
        for (int f = 3, end = (int)sqrt(n) + 1; f < end; f += 2) {
            if (n % f == 0) {
                n = 1;
                break;
            }
        }
        if (n > 1) {
            printf("%lu ", ((1ul << pow) - 1) * (1ul << (pow - 1)));
        }
    }

    elapsedTime += MPI_Wtime();

    double maxTime;

    MPI_Reduce(&elapsedTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Time elapsed: %.6f\n", maxTime);
    }
    MPI_Finalize();
    return 0;
}
