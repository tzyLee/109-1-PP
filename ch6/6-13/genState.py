import struct

state = """
    10011
    00011
    00001
    11101
    00001
"""
# state = state.replace("1", "\x01").replace("0", "\x00")
state2D = [row.strip().encode() for row in state.split("\n") if row]
M = len(state2D)
N = len(state2D[0])

with open("state", "wb+") as f:
    f.write(struct.pack("ii", M, N))
    f.writelines(state2D)
