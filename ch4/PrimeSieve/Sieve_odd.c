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

    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // if N is even => number of odds in [3, N-1]
    //              => 3 + 2(M-1) = N-1
    //              => M = (N-4)/2+1
    // if N is odd  => number of odds in [3, N]
    //              => 3 + 2(M-1) = N
    //              => M = (N-3)/2+1
    // only check **odd** numbers in [3, N], and its size is M
    const int M = N % 2 == 0 ? (N - 4) / 2 + 1 : (N - 3) / 2 + 1;
    int checkStart = 3 + 2 * BLOCK_LOW(rank, np, M);
    int checkEnd = 3 + 2 * BLOCK_HIGH(rank, np, M);
    int checkSize = BLOCK_SIZE(rank, np, M);

    // All primes should be in processor 0
    int proc0Size = M / np;
    if (3 + 2 * proc0Size < (int)sqrt((double)N)) {
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
        if (rank == 0) {
            while (marked[++searchIndex])
                ;
            // index 0 is 3 (search range is [3, N])
            prime = 2 * searchIndex + 3;
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
        ++count;  // 2 is an even prime, not in [3, N]
        printf("%d primes are less than or equal to %d\n", count, N);
    }

    MPI_Finalize();
    return 0;
}