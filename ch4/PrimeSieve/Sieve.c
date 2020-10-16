#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 100
#define BLOCK_LOW(rank, np, n) ((rank) * (n) / (np))
#define BLOCK_HIGH(rank, np, n) (BLOCK_LOW((rank) + 1, np, n) - 1)
#define BLOCK_SIZE(rank, np, n) \
    (BLOCK_HIGH(rank, np, n) - BLOCK_LOW(rank, np, n) + 1)
#define BLOCK_OWNER(index, np, n) ((np) * (((index) + 1) / (n)))

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // only check numbers in [2, N], and its size is N-1
    int checkStart = 2 + BLOCK_LOW(rank, np, N - 1);
    int checkEnd = 2 + BLOCK_HIGH(rank, np, N - 1);
    int checkSize = BLOCK_SIZE(rank, np, N - 1);

    // All primes should be in processor 0
    int proc0Size = (N - 1) / np;
    if (proc0Size + 1 < (int)sqrt((double)N)) {
        if (rank == 0) {
            printf("Too many processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    char* marked = (char*)malloc(checkSize);
    if (marked == NULL) {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        return 1;
    }

    memset(marked, 0, checkSize);

    int prime = 2;
    int searchIndex = 0;  // search unmarked number from this index
    // Mark primes
    do {
        // Find the index of smallest multiple of prime
        int firstIdx;
        if (prime * prime > checkStart) {
            firstIdx = prime * prime - checkStart;  // index of prime*prime
        } else {
            if (checkStart % prime == 0)
                // index of checkStart
                firstIdx = 0;
            else
                // index of checkStart + (prime - (checkStart % prime))
                firstIdx = prime - (checkStart % prime);
        }

        // Mark all multiple of prime
        for (int i = firstIdx; i < checkSize; i += prime) {
            marked[i] = 1;
        }
        // process 0 finds the next prime (smallest number that is not marked)
        if (rank == 0) {
            while (marked[++searchIndex])
                ;
            prime = searchIndex + 2;  // index 0 is 2 (search range is [2, N])
        }
        // Broadcast next prime to all processor
        MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } while (prime * prime <= N);

    // Count primes
    int localCount = 0, count = 0;
    for (int i = 0; i < checkSize; ++i) {
        if (!marked[i])
            ++localCount;
    }
    MPI_Reduce(&localCount, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("%d primes are less than or equal to %d\n", count, N);
    }

    MPI_Finalize();
    return 0;
}