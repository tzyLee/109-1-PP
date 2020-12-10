import numpy as np
import struct
import argparse

# Use this line to set the dtype of the array
# dtype = (int, np.int32)
dtype = (float, np.float64)


def packTuple(tup, size):
    fmt = "{}{}".format(len(tup), size)
    return struct.pack(fmt, *tup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write random ndarray into binary file."
    )

    parser.add_argument(
        "fileName",
        type=str,
        help="The destination file in which the array is written.",
    )

    parser.add_argument(
        "shape", type=int, nargs="+", help="The shape of the array to be generated."
    )

    parser.add_argument(
        "--min",
        metavar="min",
        type=dtype[0],
        default=-100,
        help="The lowest possible value in the ranodm generated array.",
    )
    parser.add_argument(
        "--max",
        metavar="max",
        type=dtype[0],
        default=100,
        help="The highest possible value in the ranodm generated array.",
    )

    args = parser.parse_args()
    print("args:", args)

    if dtype[0] is int:
        rand = np.random.randint(args.min, args.max, args.shape, dtype=dtype[1])
    elif dtype[0] is float:
        rand = (args.max - args.min) * np.random.rand(*args.shape) + args.min
    else:
        raise NotImplementedError
    with open(args.fileName, "wb") as f:
        f.write(packTuple(args.shape, "i"))
        f.write(rand.tobytes(order="C"))

    print("The array is:")
    print(rand)
    print("Array is written to {}".format(args.fileName))
