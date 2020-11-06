import sys
import re


def decompress(line, x):
    count = 0
    buf = []
    for c in line:
        if c == "b":
            # dead
            buf.append(max(count, 1) * "0")
            count = 0
        elif c == "o":
            # alive
            buf.append(max(count, 1) * "1")
            count = 0
        elif c.isdigit():
            count = 10 * count + (ord(c) - ord("0"))
    return "".join(buf).ljust(x, "0")


def decode(s, x):
    return "\n".join(decompress(line, x) for line in s.split("$"))


if __name__ == "__main__":
    assert len(sys.argv) == 2, f"Usage: python {__file__} [RLE file path]"
    fileName = sys.argv[1]

    print("Decompressing file:", fileName)

    x = 0
    with open(fileName, "r") as f:
        content = []
        for line in f:
            if line.startswith("#"):
                continue
            elif line.startswith("x"):
                m = re.match(r"x ?= ?(\d+)", line)
                x = int(m[1])
            else:
                content.append(line.strip())

    print(decode("".join(content), x))
