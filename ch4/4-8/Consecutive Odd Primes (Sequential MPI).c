#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 1000000
#define BLOCK_LOW(rank, np, n) ((rank) * (n) / (np))
#define BLOCK_HIGH(rank, np, n) (BLOCK_LOW((rank) + 1, np, n) - 1)
#define BLOCK_SIZE(rank, np, n) \
    (BLOCK_HIGH(rank, np, n) - BLOCK_LOW(rank, np, n) + 1)
#define BLOCK_OWNER(index, np, n) ((np) * (((index) + 1) / (n)))

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Status status;
    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // using only one process
    if (np != 1) {
        printf("Too many processes\n");
        MPI_Finalize();
        return 1;
    }

    const int M = N % 2 == 0 ? (N - 4) / 2 + 1 : (N - 3) / 2 + 1;
    int checkStart = 3;
    // Check additional odd number to avoid inter-process communication
    int checkEnd = 3 + 2 * (M - 1);
    int checkSize = M;

    char* marked = (char*)malloc(checkSize);
    if (marked == NULL) {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        return 1;
    }

    memset(marked, 0, checkSize);
    double elapsed_time = 0;
    elapsed_time += -MPI_Wtime();
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
        // process 0 finds the next prime (smallest number that is not
        while (marked[++searchIndex])
            ;
        // index 0 is 3 (search range is [3, N])
        prime = 2 * searchIndex + 3;
    } while (prime * prime <= N);

    // Count primes
    int localCount = 0;
    for (int i = 0; i + 1 < checkSize; ++i) {
        if (!marked[i] && !marked[i + 1])
            // both 3+2*i and 3+2*(i+1) are prime
            ++localCount;
    }
    elapsed_time += MPI_Wtime();

    printf("Number of consecutive odd prime pairs less than %d: %d\n", N,
           localCount);
    printf("Time elapsed: %f, number of processes: %d\n", elapsed_time, np);

    fflush(stdout);
    free(marked);
    MPI_Finalize();
    return 0;
}