import numpy as np


def main():
    for i in range(32):
        r = 0.5 - 0.5 * np.cos(2 * np.pi * i / 31)
        print(r)

    r = np.hanning(30)

    print(tuple(r.tolist()))


if __name__ == '__main__':
    main()
