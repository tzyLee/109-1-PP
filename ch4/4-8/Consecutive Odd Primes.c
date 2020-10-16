#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>

static bool isPrime(int n);
// Usage:
// mpicc -o [EXECUTABLE_NAME] __FILE__
// mpiexec -n [NUMBER_OF_PROCESS] [EXECUTABLE_NAME]
int main(int argc, char *argv[]) {
    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int count = 0;
    int localCount = 0;

    np *= 2;
    for (int n = 2 * rank + 1; n < 999999; n += np) {
        if (isPrime(n) && isPrime(n + 2))
            ++localCount;
    }

    MPI_Reduce(&localCount, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Number of consecutive odd prime pairs: %d\n", count);
    }
    fflush(stdout);
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
