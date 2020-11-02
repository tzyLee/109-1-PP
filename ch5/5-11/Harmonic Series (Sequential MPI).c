#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int base = 1000000000;
const int digitsOfBase = 9;
const char* digitFormat = "%09d";

int main(int argc, char* argv[]) {
    assert(argc == 3);
    int N = atoi(argv[1]);
    int d = atoi(argv[2]);

    int rank = 0;
    int np = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // using only one process
    if (np != 1) {
        printf("Too many processes\n");
        fflush(stdout);
        MPI_Finalize();
        return 1;
    }

    // d*log10(N) decimal digits after decimal point + integer part
    int len = (int)ceil(ceil(d + log10(N)) / digitsOfBase) + 1;
    int* sum = malloc(len * sizeof(int));
    memset(sum, 0, len * sizeof(int));
    double elapsed_time = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();
    for (int divisor = 1; divisor <= N; ++divisor) {
        long long dividend = 1;
        // long division
        for (int i = 0; i < len; ++i) {
            sum[i] += dividend / divisor;
            dividend %= divisor;
            dividend *= base;
        }
        // perform carrying
        for (int i = len - 1; i >= 1; --i) {
            sum[i - 1] += sum[i] / base;
            sum[i] %= base;
        }
    }
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

    free(sum);
    MPI_Finalize();
    return 0;
}

/**
 * Ref answer:
 * 14.392726722865723631381127493188587676644800013744
 * 3116534184330458129585075179950035682981759472191007..
 **/