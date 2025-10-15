from schwimmbad import MPIPool

def work(x):
    return x**2

if __name__ == "__main__":
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        results = pool.map(work, range(10))
        print(list(results))
