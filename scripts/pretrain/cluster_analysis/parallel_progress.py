from joblib import Parallel, delayed
import threading

def parallel_with_progress(delayed_func, blocks, n_jobs, print_every=50):
    """
    Run Parallel with a progress print every `print_every` blocks.
    """
    total = len(blocks)
    counter = {'count': 0}
    lock = threading.Lock()

    def wrapper(*args, **kwargs):
        result = delayed_func(*args, **kwargs)
        with lock:
            counter['count'] += 1
            i = counter['count']
            if i % print_every == 0 or i == total:
                print(f"Processed {i}/{total} day blocks...")
        return result

    results = Parallel(n_jobs=n_jobs)(delayed(wrapper)(block) for block in blocks)
    return results
