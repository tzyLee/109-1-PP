// #include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// A cylindrical hole with diameter d Is drilled completely through a cube with
// edge length s so that the center of the cylindrical hole intersects two
// opposite corners of the cube. (See Figure 10.24.) Write a program to
// determine, with five digits of precision, the volume of the portion of the
// cube that remains when s = 2 and d = 0.3.

#define BLOCK_LOW(rank, np, n) ((rank) * (n) / (np))
#define BLOCK_HIGH(rank, np, n) (BLOCK_LOW((rank) + 1, np, n) - 1)
#define BLOCK_SIZE(rank, np, n) \
    (BLOCK_HIGH(rank, np, n) - BLOCK_LOW(rank, np, n) + 1)
#define BLOCK_OWNER(index, np, n) ((np) * (((index) + 1) / (n)))

const int sideLen = 2;
const double diameter = 0.3;

int main(int argc, char *argv[]) {
    const long N_SAMPLE = atol(argv[1]);
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: ./program [N_SAMPLE]\n");
            fflush(stderr);
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsedTime = -MPI_Wtime(), maxTime;
    long start = BLOCK_LOW(rank, size, N_SAMPLE);
    long end = BLOCK_HIGH(rank, size, N_SAMPLE);

    long nRem = 0, totalRem = 0;
    srand((unsigned)time(NULL) + rank);
    const double threshold = 3 * (diameter / 2) * (diameter / 2);
    for (long i = start; i <= end; ++i) {
        double x = (double)rand() / RAND_MAX * sideLen;
        double y = (double)rand() / RAND_MAX * sideLen;
        double z = (double)rand() / RAND_MAX * sideLen;
        // double diag = sqrt(x * x + y * y + z * z);
        // if (sin(acos((x + y + z) / (sqrt3 * diag))) * diag > radius)
        if ((y - z) * (y - z) + (z - x) * (z - x) + (x - y) * (x - y) >
            threshold) {
            // The point is not removed
            ++nRem;
        }
    }

    MPI_Reduce(&nRem, &totalRem, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    long double volume = 0;
    if (rank == 0) {
        volume = sideLen * sideLen * sideLen * (double)(totalRem / N_SAMPLE);
    }
    elapsedTime += MPI_Wtime();

    MPI_Reduce(&elapsedTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        printf(
            "The volume is %.6Lf (Number of sample = %ld), time elapsed = "
            "%.6f (Number of process = %d)\n",
            volume, N_SAMPLE, maxTime, size);
        fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}
