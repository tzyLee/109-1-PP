#include <mpi.h>
#include <stdio.h>

#define N 50

double f(int i);

// Usage:
// mpicc -o [EXECUTABLE_NAME] __FILE__
// mpiexec -n [NUMBER_OF_PROCESS] [EXECUTABLE_NAME]
int main(int argc, char *argv[]) {
    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    double elapsed_time = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();

    double total_area = 0;
    double area = 0;

    for (int i = 2 * (rank + 1); i <= N; i += 2 * np)
        area += 4.0 * f(i - 1) + 2.0 * f(i);

    MPI_Reduce(&area, &total_area, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    elapsed_time += MPI_Wtime();
    if (rank == 0) {
        total_area += f(0) - f(N);
        total_area /= 3.0 * N;
        printf("Approximation of pi: %13.11f\n", total_area);
        printf("Time elapsed: %f, number of processes: %d\n", elapsed_time, np);
        fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}

double f(int i) {
    double x = (double)i / N;
    return 4.0 / (1.0 + x * x);
}
