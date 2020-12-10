import numpy as np
import struct


with open("matrix", "rb") as m, open("vector", "rb") as v:
    mShape = struct.unpack("ii", m.read(8))
    vShape = struct.unpack("i", v.read(4))

    matrix = np.frombuffer(m.read()).reshape(mShape)
    vector = np.frombuffer(v.read()).reshape(vShape)

    product = matrix @ vector

    print("product is:")
    print(product)
