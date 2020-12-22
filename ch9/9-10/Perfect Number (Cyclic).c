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

static int cmpInt(const void* a, const void* b) {
    return *(unsigned long*)a == *(unsigned long*)b
               ? 0
               : *(unsigned long*)a > *(unsigned long*)b
                     ? 1
                     : -1;  // a-b might overflow
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        fprintf(stderr, "This program needs at least 2 processes to work.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsedTime = -MPI_Wtime();

    unsigned long perfectNumbers[N + 1] = {0};
    if (rank == 0) {
        --size;
        MPI_Status status;
        int wInd = 0, nRecv = 0;
        for (int nRunning = size; nRunning > 0;) {
            MPI_Recv(&perfectNumbers[wInd], 1, MPI_UNSIGNED_LONG,
                     MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_UNSIGNED_LONG, &nRecv);
            if (nRecv == 0)
                --nRunning;
            else
                ++wInd;
        }
        qsort(perfectNumbers, N, sizeof(unsigned long), cmpInt);
    } else {
        rank %= size;
        --size;
        MPI_Request req;
        unsigned long perfectNumber;
        for (int pow = rank; pow < MAX_POW; pow += size) {
            unsigned long n = (1ul << pow) - 1;
            for (int f = 3, end = (int)sqrt(n) + 1; f < end; f += 2) {
                if (n % f == 0) {
                    n = 1;
                    break;
                }
            }
            if (n > 1) {
                perfectNumber = ((1ul << pow) - 1) * (1ul << (pow - 1));
                MPI_Send(&perfectNumber, 1, MPI_UNSIGNED_LONG, 0, 0,
                         MPI_COMM_WORLD);
            }
        }
        MPI_Isend(NULL, 0, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, &req);
    }
    elapsedTime += MPI_Wtime();

    double maxTime;

    MPI_Reduce(&elapsedTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Perfect numbers: [");
        for (int i = 0; i < N; ++i) {
            printf("%lu, ", perfectNumbers[i]);
        }
        printf("], Time elapsed: %.6f\n", maxTime);
    }
    MPI_Finalize();
    return 0;
}
