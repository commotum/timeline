#!/usr/bin/env python3
import csv
import random


def main() -> None:
    values = random.sample(range(16**6), 400)
    values.sort()
    hex_strings = [format(value, "06x") for value in values]

    with open("id.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows([[hex_string] for hex_string in hex_strings])


if __name__ == "__main__":
    main()
