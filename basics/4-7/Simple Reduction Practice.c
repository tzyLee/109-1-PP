#include <mpi.h>
#include <stdio.h>

// Usage:
// mpicc -o [EXECUTABLE_NAME] __FILE__
// mpiexec -n [NUMBER_OF_PROCESS] [EXECUTABLE_NAME]
int main(int argc, char *argv[]) {
    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int sum = 0;
    int term = rank + 1;

    MPI_Reduce(&term, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        // prevents overflow when calculating `np*(np+1)/2`
        int ref_answer =
            (np % 2 == 0) ? (np / 2) * (np + 1) : ((np + 1) / 2) * np;
        printf(
            "Number of processes: %d, "
            "Reduction result: %d, "
            "Reference answer: %d\n",
            np, sum, ref_answer);
    }
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
