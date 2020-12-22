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
#define BUF_SIZE 50

enum MessageTag {
    EMPTY_MSG,  /* The message that is sent as request */
    RESULT_MSG, /* The computation result, sends N if the number is perfect.
                   otherwise, sends -N */
    N_MSG       /* The parameter of computation */
};

enum Status { UNDEF, PERFECT, NOT_PERFECT };

void manager(int rank, int size);
void worker(int rank, int size, MPI_Comm commWorker);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        fprintf(stderr, "This program needs at least 2 processes to run\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm commWorker;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, rank, &commWorker);
        manager(rank, size);
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &commWorker);
        worker(rank, size, commWorker);
    }

    MPI_Finalize();
    return 0;
}

void manager(int rank, int size) {
    unsigned char isPerfect[BUF_SIZE] = {0};
    isPerfect[1] = isPerfect[0] = NOT_PERFECT;  // 0, 1 is not a prime
    int lastN = 1;     // N = 0, ... lastN is already computed
    int nPerfect = 0;  // Number of perfect number seen in N = 0, ..., lastN
    int result;
    int nRunning = size - 1;
    MPI_Status status;

    double elapsedTime = -MPI_Wtime();
    for (int pow = 2; nRunning > 0; ++pow) {
        MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == RESULT_MSG) {
            int n;
            if (result > 0) {
                isPerfect[result] = PERFECT;
                n = result;
            } else {
                isPerfect[-result] = NOT_PERFECT;
                n = -result;
            }
            for (int i = lastN + 1; i <= n; ++i) {
                if (isPerfect[i] == UNDEF)
                    break;
                else if (isPerfect[i] == PERFECT)
                    ++nPerfect;
                lastN = i;
            }
        }
        if (nPerfect < N && pow < BUF_SIZE) {
            MPI_Send(&pow, 1, MPI_INT, status.MPI_SOURCE, N_MSG,
                     MPI_COMM_WORLD);
        } else {
            MPI_Send(NULL, 0, MPI_INT, status.MPI_SOURCE, N_MSG,
                     MPI_COMM_WORLD);
            --nRunning;
        }
    }
    elapsedTime += MPI_Wtime();

    printf("Found perfect numbers: [");
    for (int i = 0, rem = N; i < BUF_SIZE; ++i) {
        if (isPerfect[i] == PERFECT) {
            printf("%lu, ", ((1lu << i) - 1) * (1lu << (i - 1)));
            if (--rem == 0)
                break;
        }
    }
    printf("], ");
    printf("Time elapsed: %.6f\n", elapsedTime);
    fflush(stdout);
}

void worker(int rank, int size, MPI_Comm commWorker) {
    int pow;
    unsigned long n = 0;
    MPI_Status status;
    MPI_Send(NULL, 0, MPI_INT, 0, EMPTY_MSG, MPI_COMM_WORLD);
    while (1) {
        MPI_Recv(&pow, 1, MPI_INT, 0, N_MSG, MPI_COMM_WORLD, &status);
        int recvCnt;
        MPI_Get_count(&status, MPI_UNSIGNED_LONG, &recvCnt);
        if (recvCnt == 0)
            break;
        n = ((1ul << pow) - 1);
        for (int i = 3, end = (int)sqrt(n) + 1; i < end; i += 2) {
            if (n % i == 0) {
                pow = -pow;  // send -pow back (not perfect)
                break;
            }
        }
        // if (pow > 0)
        //     printf("Process %d: (2^%d - 1) is prime\n", rank, pow);
        // else
        //     printf("Process %d: (2^%d - 1) is not prime\n", rank, -pow);
        // fflush(stdout);
        MPI_Send(&pow, 1, MPI_INT, 0, RESULT_MSG, MPI_COMM_WORLD);
    }
}