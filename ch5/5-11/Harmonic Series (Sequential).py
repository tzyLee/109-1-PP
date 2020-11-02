import math
from fractions import Fraction


def main():
    N = int(input("Number of terms to sum: "))
    D = int(input("N digits of precision: "))
    D += int(math.ceil(math.log10(N)))

    print(f"N = {N}, D = {D}")

    harmonicSum = Fraction()
    for i in range(1, N + 1):
        harmonicSum += Fraction(1, i)

    print(harmonicSum)


if __name__ == "__main__":
    main()
