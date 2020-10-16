#include <mpi.h>
#include <stdio.h>

// Usage:
// mpicc -o [EXECUTABLE_NAME] __FILE__
// mpiexec -n [NUMBER_OF_PROCESS] [EXECUTABLE_NAME]
int main(int argc, char *argv[]) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("hello, world, from process <%d>\n", rank);
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
