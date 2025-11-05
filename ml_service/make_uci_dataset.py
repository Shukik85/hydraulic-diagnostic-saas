#!/usr/bin/env python3
"""
Make UCI dataset from RAW and export tidy CSV.
"""
from uci_raw_loader import build_raw_dataset

if __name__ == "__main__":
    r = build_raw_dataset()
    print(r)
