#include <assert.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_LOW(rank, np, n) ((rank) * (n) / (np))
#define BLOCK_HIGH(rank, np, n) (BLOCK_LOW((rank) + 1, np, n) - 1)
#define BLOCK_SIZE(rank, np, n) \
    (BLOCK_HIGH(rank, np, n) - BLOCK_LOW(rank, np, n) + 1)
#define BLOCK_OWNER(index, np, n) ((np) * (((index) + 1) / (n)))

// use 4bit msb to store the new state
const unsigned MASK = 0xF;
const unsigned SHIFT = 4;
#define SET_LIVE(x)          \
    do {                     \
        (x) |= 1 << (SHIFT); \
    } while (0)
#define SET_DEAD(x)             \
    do {                        \
        (x) &= ~(1 << (SHIFT)); \
    } while (0)
#define UPDATE(x, liveCount)                            \
    do {                                                \
        if ((x)&1) {                                    \
            if ((liveCount) == 2 || (liveCount) == 3) { \
                SET_LIVE((x));                          \
            } else {                                    \
                SET_DEAD((x));                          \
            }                                           \
        } else {                                        \
            if ((liveCount) == 3) {                     \
                SET_LIVE((x));                          \
            } else {                                    \
                SET_DEAD((x));                          \
            }                                           \
        }                                               \
    } while (0)

void readRowMajorMatrix(char* fileName, char** array, int* M, int* N,
                        MPI_Comm comm) {
    FILE* inFile = NULL;
    MPI_Status status;
    int np, rank;
    MPI_Comm_size(comm, &np);
    MPI_Comm_rank(comm, &rank);

    if (rank == np - 1) {
        inFile = fopen(fileName, "r");
        if (inFile == NULL) {
            MPI_Abort(comm, -1);
            return;
        }
        fread(M, sizeof(int), 1, inFile);
        fread(N, sizeof(int), 1, inFile);
    }

    MPI_Bcast(M, 1, MPI_INT, np - 1, comm);
    MPI_Bcast(N, 1, MPI_INT, np - 1, comm);

    int chunkSize = BLOCK_SIZE(rank, np, *M) * (*N);
    // +2N to store a row before and a row after the block
    *array = malloc((chunkSize + 2 * (*N)) * sizeof(char));
    if (rank == 0) {
        memset(*array, '0', (*N) * sizeof(char));
    } else if (rank == np - 1) {
        memset(*array + chunkSize + (*N), '0', (*N) * sizeof(char));
    }
    // +*N to reserve an empty row
    char* blockStart = *array + *N;
    if (rank == np - 1) {
        for (int i = 0; i < np - 1; ++i) {
            int otherChunkSize = BLOCK_SIZE(i, np, *M) * (*N);
            fread(blockStart, sizeof(char), otherChunkSize, inFile);
            MPI_Send(blockStart, otherChunkSize, MPI_CHAR, i, 0, comm);
        }
        fread(blockStart, sizeof(char), chunkSize, inFile);
        fclose(inFile);
    } else {
        MPI_Recv(blockStart, chunkSize, MPI_CHAR, np - 1, 0, comm, &status);
    }
}

char** reshapeTo2D(char* array, int M, int N) {
    char** reshaped = malloc(M * sizeof(char*));
    for (int i = 0; i < M; ++i) {
        reshaped[i] = array;
        array += N;
    }
    return reshaped;
}

int main(int argc, char* argv[]) {
    assert(argc == 3);
    const int ITERATIONS = atoi(argv[1]);
    const int PRINT_CYCLE = atoi(argv[2]);

    MPI_Init(&argc, &argv);
    int np, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int M = 0, N = 0;
    char* flattenedArray = NULL;
    char* recvBuffer = NULL;
    if (rank == 0) {
        recvBuffer = malloc(BLOCK_SIZE(np - 1, np, M) * N * sizeof(char));
    }
    readRowMajorMatrix("state", &flattenedArray, &M, &N, MPI_COMM_WORLD);

    int localM = BLOCK_SIZE(rank, np, M);
    // +2 for row before and after block
    char** array = reshapeTo2D(flattenedArray, localM + 2, N);
    MPI_Status status;
    for (int i = 0; i < ITERATIONS; ++i) {
        if (rank - 1 >= 0) {
            // p send first row to p-1
            MPI_Send(array[1], N, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD);
        }
        if (rank + 1 < np) {
            // p recv first row from p+1
            MPI_Recv(array[localM + 1], N, MPI_CHAR, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
            // p send last row to p+1
            MPI_Send(array[localM], N, MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD);
        }
        if (rank - 1 >= 0) {
            // p recv last row from p-1
            MPI_Recv(array[0], N, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD,
                     &status);
        }
        // calculate new state of each cell
        for (int r = 1; r <= localM; ++r) {
            int liveCount = (array[r - 1][0] + array[r - 1][1] + array[r][0] +
                             array[r][1] + array[r + 1][0] + array[r + 1][1]) &
                            MASK;
            UPDATE(array[r][0], liveCount - (array[r][0] & MASK));
            for (int c = 1; c < N - 1; ++c) {
                // add col
                liveCount += (array[r - 1][c + 1] + array[r][c + 1] +
                              array[r + 1][c + 1]) &
                             MASK;
                UPDATE(array[r][c], liveCount - (array[r][c] & MASK));
                // remove col
                liveCount -= (array[r - 1][c - 1] + array[r][c - 1] +
                              array[r + 1][c - 1]) &
                             MASK;
            }
            UPDATE(array[r][N - 1], liveCount - (array[r][N - 1] & MASK));
        }
        for (int r = 1; r <= localM; ++r) {
            for (int c = 0; c < N; ++c) {
                // set to new state
                array[r][c] = ((array[r][c] & (1 << SHIFT)) >> SHIFT) | '0';
            }
        }
        if (i % PRINT_CYCLE == 0) {
            if (rank != 0) {
                MPI_Send(array[1], localM * N, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
            } else {
                for (int j = 1; j <= localM; ++j) {
                    fwrite(array[j], 1, N, stdout);
                    putchar('\n');
                }
                for (int i = 1; i < np; ++i) {
                    int localM = BLOCK_SIZE(i, np, M);
                    MPI_Recv(recvBuffer, localM * N, MPI_CHAR, i, 1,
                             MPI_COMM_WORLD, &status);
                    char* temp = recvBuffer;
                    for (int j = 0; j < localM; ++j) {
                        fwrite(temp, 1, N, stdout);
                        putchar('\n');
                        temp += N;
                    }
                }
                putchar('\n');
                fflush(stdout);
            }
        }
    }

    if (rank == 0) {
        free(recvBuffer);
    }
    free(array);
    free(flattenedArray);
    MPI_Finalize();
    return 0;
}