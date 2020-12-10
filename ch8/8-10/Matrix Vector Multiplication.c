#include <errno.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double dtype;
#define DTYPE MPI_DOUBLE
#define FMT "%.8f"

// Write a program that implements matrix-vector multiplication based on a
// checkerboard biock decomposition of the matrix.
// The program should read the matrix and the vector from an input file and
// print the answer to standard output.
// The names of the files containing the matrix and the vector should be
// specified as command-line arguments.

#define BLOCK_LOW(rank, np, n) ((rank) * (n) / (np))
#define BLOCK_HIGH(rank, np, n) (BLOCK_LOW((rank) + 1, np, n) - 1)
#define BLOCK_SIZE(rank, np, n) \
    (BLOCK_HIGH(rank, np, n) - BLOCK_LOW(rank, np, n) + 1)
#define BLOCK_OWNER(index, np, n) ((np) * (((index) + 1) / (n)))

void* safeMalloc(int rank, size_t size);
int safeFread(void* ptr, size_t size, size_t nmemb, FILE* stream);
void readCheckerboardMatrix(const char* filename, dtype** array, int* M, int* N,
                            MPI_Comm comm);
void readBlockVector(const char* filename, dtype** array, int* len,
                     MPI_Comm comm);
void writeBlockVector(FILE* stream, dtype* array, int len, MPI_Comm comm);
void writeArray(FILE* stream, dtype* array, int len);

int errnum = 1;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s [matrix-file] [vector-file]\n",
                (argc != 0) ? argv[0] : "./mvm");
        return 1;
    }
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create cartesian communicator
    int shape[2] = {0};
    int periodic[2] = {0};
    MPI_Comm commGrid, commRow, commCol;
    MPI_Dims_create(size, 2, shape);
    MPI_Cart_create(MPI_COMM_WORLD, 2, shape, periodic, 1, &commGrid);

    // Split into col and row communicator groups
    int coords[2] = {0};
    MPI_Cart_coords(commGrid, rank, 2, coords);
    MPI_Comm_split(commGrid, coords[0], coords[1], &commRow);
    MPI_Comm_split(commGrid, coords[1], coords[0], &commCol);

    dtype *flattenMatrix = NULL, *vector = NULL;
    int M, N, vecSize;
    // Read matrix from file and split into blocks
    readCheckerboardMatrix(argv[1], &flattenMatrix, &M, &N, commGrid);

    // Split vector into blocks and send to subgrids on the first row
    if (coords[0] == 0) {
        readBlockVector(argv[2], &vector, &vecSize, commRow);
    }
    // Broadcast vector size and vector from first block of each column
    MPI_Bcast(&vecSize, 1, MPI_INT, 0, commCol);
    if (N != vecSize) {
        fprintf(stderr,
                "The matrix cannot be multiplied with the given vector\n");
        MPI_Abort(MPI_COMM_WORLD, EINVAL);
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsedTime = -MPI_Wtime();
    int blockSize = BLOCK_SIZE(coords[1], shape[1], vecSize);
    if (coords[0] != 0) {
        vector = safeMalloc(rank, blockSize * sizeof(dtype));
    }
    MPI_Bcast(vector, blockSize, DTYPE, 0, commCol);

    // Multiply each block
    int blockM = BLOCK_SIZE(coords[0], shape[0], M);
    int blockN = BLOCK_SIZE(coords[1], shape[1], N);
    dtype* partialProduct = safeMalloc(rank, blockM * sizeof(dtype));
    memset(partialProduct, 0, blockM * sizeof(dtype));
    dtype* mPtr = flattenMatrix;
    for (int i = 0; i < blockM; ++i) {
        for (int j = 0; j < blockN; ++j) {
            partialProduct[i] += *mPtr * vector[j];
            ++mPtr;
        }
    }

    // Sum up partial products and store in the first column
    dtype* product = NULL;
    if (coords[1] == 0) {  // first column
        product = safeMalloc(rank, blockM * sizeof(dtype));
    }
    MPI_Reduce(partialProduct, product, blockM, DTYPE, MPI_SUM, 0, commRow);
    if (coords[1] == 0) {
        writeBlockVector(stdout, product, blockM, commCol);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    elapsedTime += MPI_Wtime();

    double maxElapsedTime;
    MPI_Reduce(&elapsedTime, &maxElapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\nNumber of process=%d, Time elapsed=%.6f sec(s)\n", size,
               maxElapsedTime);
        fflush(stdout);
    }
    if (coords[1] == 0) {
        free(product);
        product = NULL;
    }
    free(partialProduct);
    partialProduct = NULL;
    free(flattenMatrix);
    flattenMatrix = NULL;
    free(vector);
    vector = NULL;
    MPI_Finalize();
    return 0;
}

void* safeMalloc(int rank, size_t size) {
    void* buf = NULL;
    if ((buf = malloc(size)) == NULL) {
        errnum = errno;
        perror("malloc failed");
        MPI_Abort(MPI_COMM_WORLD, errnum);
    }
    return buf;
}

int safeFread(void* ptr, size_t size, size_t nmemb, FILE* stream) {
    int nRead = -1;
    if ((nRead = fread(ptr, size, nmemb, stream)) < 0) {
        errnum = errno;
        perror("fread failed");
        MPI_Abort(MPI_COMM_WORLD, errnum);
    }
    return nRead;
}

void readCheckerboardMatrix(const char* filename, dtype** array, int* M, int* N,
                            MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    FILE* inFile = NULL;
    *M = *N = 0;
    if (rank == size - 1) {
        if ((inFile = fopen(filename, "rb")) == NULL) {
            errnum = errno;
            perror("fopen failed");
            MPI_Abort(MPI_COMM_WORLD, errnum);
            return;
        }
        safeFread(M, sizeof(int), 1, inFile);
        safeFread(N, sizeof(int), 1, inFile);
    }
    MPI_Bcast(M, 1, MPI_INT, size - 1, comm);
    if (*M == 0) {
        MPI_Abort(MPI_COMM_WORLD, ENOENT);
        return;
    }
    MPI_Bcast(N, 1, MPI_INT, size - 1, comm);

    int shape[2], coords[2], periodic[2];
    MPI_Cart_get(comm, 2, shape, periodic, coords);
    int blockM = BLOCK_SIZE(coords[0], shape[0], *M);
    int blockN = BLOCK_SIZE(coords[1], shape[1], *N);
    dtype* const _array = safeMalloc(rank, blockM * blockN * sizeof(dtype));
    *array = _array;

    dtype* buffer = NULL;
    unsigned delta = 0;
    int destCoords[2], destRank;
    if (rank == size - 1) {
        buffer = safeMalloc(rank, *N * sizeof(dtype));
        for (int i = 0; i < shape[0]; ++i) {
            const int iBlockM = BLOCK_SIZE(i, shape[0], *M);
            destCoords[0] = i;
            for (int rowCount = 0; rowCount < iBlockM; ++rowCount) {
                safeFread(buffer, sizeof(dtype), *N, inFile);
                delta = 0;
                for (int j = 0; j < shape[1]; ++j) {
                    destCoords[1] = j;
                    const int iBlockN = BLOCK_SIZE(j, shape[1], *N);
                    MPI_Cart_rank(comm, destCoords, &destRank);
                    if (destRank != rank) {  // send to destRank (coords=[i, j])
                        MPI_Send(buffer + delta, iBlockN, DTYPE, destRank, 0,
                                 comm);
                    } else {  // copy to itself
                        memcpy(_array + rowCount * iBlockN, buffer + delta,
                               iBlockN * sizeof(dtype));
                    }
                    delta += iBlockN;
                }
            }
        }
        free(buffer);
        buffer = NULL;
    } else {
        delta = 0;
        for (int i = 0; i < blockM; ++i) {
            MPI_Recv(_array + delta, blockN, DTYPE, size - 1, 0, comm,
                     MPI_STATUS_IGNORE);
            delta += blockN;
        }
    }
}

// Read a vector from `filename` and divide it into blocks among processes in a
// communicator group.
void readBlockVector(const char* filename, dtype** array, int* len,
                     MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    FILE* inFile = NULL;
    *len = 0;
    if (rank == size - 1) {
        if ((inFile = fopen(filename, "rb")) == NULL) {
            // Abort all processes in comm
            errnum = errno;
            perror("fopen failed");
            MPI_Abort(MPI_COMM_WORLD, errno);
            return;
        }
        safeFread(len, sizeof(int), 1, inFile);
    }
    MPI_Bcast(len, 1, MPI_INT, size - 1, comm);
    if (*len == 0) {
        MPI_Abort(MPI_COMM_WORLD, ENOENT);
        return;
    }

    int blockSize = BLOCK_SIZE(rank, size, *len);
    dtype* _array = safeMalloc(rank, blockSize * sizeof(dtype));
    *array = _array;
    if (rank == size - 1) {
        for (int i = 0; i < rank; ++i) {
            const int iBlockSize = BLOCK_SIZE(i, size, *len);
            safeFread(_array, sizeof(dtype), iBlockSize, inFile);
            MPI_Send(_array, iBlockSize, DTYPE, i, 0, comm);
        }
        safeFread(_array, sizeof(dtype), blockSize, inFile);
        fclose(inFile);
    } else {
        MPI_Recv(_array, BLOCK_SIZE(rank, size, *len), DTYPE, size - 1, 0, comm,
                 MPI_STATUS_IGNORE);
    }
}

// Write a block vector to stream, with each block joined according the rank of
// process, and the element is separated by ", "
void writeBlockVector(FILE* stream, dtype* array, int len, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int printSignal = 0;
    if (rank != 0) {
        MPI_Recv(&printSignal, 1, MPI_INT, rank - 1, 0, comm,
                 MPI_STATUS_IGNORE);
    }
    writeArray(stream, array, len);
    fflush(stream);
    if (rank + 1 < size) {
        if (len > 0) {
            fprintf(stream, ", ");
            fflush(stream);
        }
        MPI_Send(&printSignal, 1, MPI_INT, rank + 1, 0, comm);
    } else {
        fputc('\n', stream);
        fflush(stream);
    }
}

// Write an array to stream, separated by ", "
void writeArray(FILE* stream, dtype* array, int len) {
    if (len == 0)
        return;
    fprintf(stream, FMT, array[0]);
    for (int i = 1; i < len; ++i) {
        fprintf(stream, ", " FMT, array[i]);
    }
}