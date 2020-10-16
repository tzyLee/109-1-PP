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
    double elapsed_time = 0;
    MPI_Barrier(MPI_COMM_WORLD);
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
    for (int i = 0; i + 1 < checkSize; ++i) {
        if (!marked[i] && !marked[i + 1])
            // both 3+2*i and 3+2*(i+1) are prime
            ++localCount;
    }

    if (rank != 0) {
        MPI_Send(marked, 1, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD);
    }

    int firstMarkInNextProcess = 0;
    if (rank != np - 1) {
        MPI_Recv(&firstMarkInNextProcess, 1, MPI_CHAR, rank + 1, 0,
                 MPI_COMM_WORLD, &status);
        // check the last odd in this process
        // and the first odd in the next process
        if (!marked[checkSize - 1] && !firstMarkInNextProcess)
            ++localCount;
    }

    MPI_Reduce(&localCount, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    elapsed_time += MPI_Wtime();

    if (rank == 0) {
        printf("Number of consecutive odd prime pairs less than %d: %d\n", N,
               count);
    }
    if (np > 1 && rank == 1 || np == 1 && rank == 0) {
        // print time on process 1 if possible
        // (process 0 does not send to other)
        printf("Time elapsed: %f, number of processes: %d\n", elapsed_time, np);
    }

    fflush(stdout);
    free(marked);
    MPI_Finalize();
    return 0;
}