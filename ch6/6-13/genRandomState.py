import struct
import random

M = 1000
N = 1000

if __name__ == "__main__":
    with open("state", "wb+") as f:
        f.write(struct.pack("ii", M, N))
        rand = random.getrandbits(M * N)
        randBits = "{:b}".format(rand).zfill(M * N).encode()
        f.write(randBits)
