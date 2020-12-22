def isPrime(num):
    if num < 2:
        return False
    if num % 2 == 0:
        return num == 2
    check = 0
    for factor in range(3, int(num**0.5 + 1), 2):
        check += 1
        if num % factor == 0:
            return False
    return True

if __name__ == "__main__":
    found = 0
    perfects = []
    twoPowerP = 2
    while found < 9:
        if isPrime(twoPowerP -1):
            found += 1
            perfects.append(twoPowerP//2 * (twoPowerP - 1))
        twoPowerP *= 2
    
    print('found perfect numbers:', perfects)