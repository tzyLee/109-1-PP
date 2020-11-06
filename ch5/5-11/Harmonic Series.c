#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_LOW(rank, np, n) ((rank) * (n) / (np))
#define BLOCK_HIGH(rank, np, n) (BLOCK_LOW((rank) + 1, np, n) - 1)
#define BLOCK_SIZE(rank, np, n) \
    (BLOCK_HIGH(rank, np, n) - BLOCK_LOW(rank, np, n) + 1)

const int base = 1000000000;
const int digitsOfBase = 9;
const char* digitFormat = "%09d";

void bigIntAdd(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype) {
    int *in = (int*)invec, *inout = (int*)inoutvec;
    // addition and carrying
    for (int i = *len - 1; i >= 1; --i) {
        inout[i] += in[i];
        inout[i - 1] += inout[i] / base;
        inout[i] %= base;
    }
    inout[0] += in[0];
}

int main(int argc, char* argv[]) {
    assert(argc == 3);
    int N = atoi(argv[1]);
    int d = atoi(argv[2]);

    int rank = 0;
    int np = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int start = BLOCK_LOW(rank, np, N) + 1;
    int end = BLOCK_HIGH(rank, np, N) + 1;
    // d*log10(N) decimal digits after decimal point + integer part
    int len = (int)ceil(ceil(d + log10(N)) / digitsOfBase) + 1;
    int* sum = malloc(len * sizeof(int));
    int* partialSum = malloc(len * sizeof(int));
    memset(sum, 0, len * sizeof(int));
    memset(partialSum, 0, len * sizeof(int));
    double elapsed_time = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();
    for (int divisor = start; divisor <= end; ++divisor) {
        long long dividend = 1;
        // long division
        for (int i = 0; i < len; ++i) {
            partialSum[i] += dividend / divisor;
            dividend %= divisor;
            // divisor must <= LLONG_MAX/base, or dividend*base > LLONG_MAX
            dividend *= base;
        }
        // perform carrying
        for (int i = len - 1; i >= 1; --i) {
            partialSum[i - 1] += partialSum[i] / base;
            partialSum[i] %= base;
        }
    }

    MPI_Op BIGINT_ADD;
    MPI_Op_create(bigIntAdd, 1 /* is commutative */, &BIGINT_ADD);
    MPI_Reduce(partialSum, sum, len, MPI_INT, BIGINT_ADD, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        elapsed_time += MPI_Wtime();
        char buf[2048] = {0};
        char *bufStart = buf, *bufEnd = buf + sizeof(buf);
        bufStart += snprintf(bufStart, bufEnd - bufStart, "%d.", sum[0]);
        int fractionStart = bufStart - buf;
        for (int i = 1; i < len; ++i) {
            bufStart +=
                snprintf(bufStart, bufEnd - bufStart, digitFormat, sum[i]);
        }
        // ignore extra digits
        buf[fractionStart + d] = '\0';
        printf("Time elapsed: %f\nAnswer: %s\n", elapsed_time, buf);
        fflush(stdout);
    }

    MPI_Op_free(&BIGINT_ADD);
    free(sum);
    free(partialSum);
    MPI_Finalize();
    return 0;
}

/**
 * Ref answer:
 * 14.392726722865723631381127493188587676644800013744
 * 3116534184330458129585075179950035682981759472191007..
 **/