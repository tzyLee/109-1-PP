#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>

// Usage:
// mpicc -o [EXECUTABLE_NAME] __FILE__
// mpiexec -n [NUMBER_OF_PROCESS] [EXECUTABLE_NAME]
#define SAMPLE_SIZE 10000
#define MESG_START 512
#define BUFFER_SIZE 134217728

int main(int argc, char *argv[]) {
    char buf[BUFFER_SIZE] = {0}, recv_buf[BUFFER_SIZE] = {0};
    int rank, np;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    assert(np == 2);

    if (rank == 0)
        memset(buf, 'a', BUFFER_SIZE);

    for (int mesg_len = MESG_START; mesg_len <= BUFFER_SIZE; mesg_len *= 2) {
        double elapsed_time = 0;
        for (int sample = 0; sample < SAMPLE_SIZE; ++sample) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0) {
                elapsed_time += -MPI_Wtime();
                MPI_Send(buf, mesg_len, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(recv_buf, mesg_len, MPI_CHAR, 1, 1, MPI_COMM_WORLD,
                         &status);
                elapsed_time += MPI_Wtime();
            } else {
                MPI_Recv(buf, mesg_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD,
                         &status);
                MPI_Send(buf, mesg_len, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
            }
        }
        if (rank == 0) {
            // printf("Averge message passing time of sending %d bytes is:
            // %f\n", mesg_len, elapsed_time / 2 / SAMPLE_SIZE);
            printf("%d, %f\n", mesg_len, elapsed_time / 2 / SAMPLE_SIZE);
        }
    }

    fflush(stdout);
    MPI_Finalize();
    return 0;
}
