# Python Code Optimization Checklist

This checklist summarizes the primary optimization techniques for improving processing and execution speed of Python code. Use it for code reviews or self-assessment.

## 1. Algorithm and Data Structure Optimization

- **Computational Complexity Optimization**
  - Consider replacing algorithms with more efficient ones (e.g., O(N²) → O(N log N) or O(N))
  - Avoid unnecessary repeated calculations

- **Selecting Optimal Data Structures**
  - Use dictionaries (dict) or sets instead of lists for frequent searches
  ```python
  # Slow: O(n) complexity
  if x in my_list:
      # processing
  
  # Fast: O(1) complexity
  if x in my_set:  # or my_dict
      # processing
  ```
  
  - Consider `collections.deque` over `list` for frequent additions/removals at the beginning/end
  ```python
  from collections import deque
  queue = deque()
  queue.append(item)    # add to end
  queue.appendleft(item)  # add to beginning
  first_item = queue.popleft()  # remove from beginning
  ```
  
  - Use `tuple` instead of `list` for immutable sequences (memory/speed benefits)

- **Prioritize Local Variables**
  - Access to local variables is faster than global or class variables
  ```python
  # Slow
  global_var = 0
  def slow_function():
      global global_var
      for i in range(1000000):
          global_var += i
          
  # Fast
  def fast_function():
      local_var = 0
      for i in range(1000000):
          local_var += i
      return local_var
  ```

## 2. Basic Code Optimization

- **Loop Optimization**
  - Move operations that don't need to be executed in each iteration outside the loop
  ```python
  # Slow
  for i in range(1000):
      expensive_setup()  # same setup executed every time
      process(i)
  
  # Fast
  expensive_setup()  # executed only once
  for i in range(1000):
      process(i)
  ```

- **Utilize List Comprehensions**
  - They operate faster than traditional loops
  ```python
  # Slow
  result = []
  for i in range(1000):
      if i % 2 == 0:
          result.append(i * i)
          
  # Fast
  result = [i * i for i in range(1000) if i % 2 == 0]
  ```

- **Use Generator Expressions**
  - Improves memory efficiency when dealing with large amounts of data
  ```python
  # High memory consumption
  sum([x * x for x in range(1000000)])
  
  # Memory efficient
  sum(x * x for x in range(1000000))
  ```

- **Use `join` for Multiple String Concatenations**
  - String concatenation with `+` or `+=` is inefficient
  ```python
  # Slow
  s = ""
  for i in range(1000):
      s += str(i)
  
  # Fast
  s = "".join(str(i) for i in range(1000))
  ```

- **Avoid Unnecessary Object Creation**
  - Minimize unnecessary object creation within loops

## 3. Leveraging Built-in Functions and Standard Library

- **Use Built-in Functions**
  - Built-in functions like `map`, `filter`, `sum`, `min`, `max` are optimized
  ```python
  # Slow
  total = 0
  for num in numbers:
      total += num
      
  # Fast
  total = sum(numbers)
  ```

- **Utilize the `itertools` Module**
  - Provides functions for efficient iteration
  ```python
  import itertools
  
  # Efficient infinite loop
  for i in itertools.count():
      if condition:
          break
  
  # Efficient combinations
  combinations = list(itertools.combinations(items, 2))
  ```

- **Use Special Containers from the `collections` Module**
  - Appropriately use `defaultdict`, `Counter`, `namedtuple`, etc.
  ```python
  from collections import Counter
  
  # Count occurrences of elements
  word_counts = Counter(['apple', 'banana', 'apple', 'orange'])
  # Counter({'apple': 2, 'banana': 1, 'orange': 1})
  ```

- **Consider Function Call Overhead**
  - Consolidate processing rather than calling many small functions
  - Consider the `@functools.lru_cache` decorator for frequently called functions
  ```python
  from functools import lru_cache
  
  @lru_cache(maxsize=None)
  def fibonacci(n):
      if n < 2:
          return n
      return fibonacci(n-1) + fibonacci(n-2)
  ```

## 4. Memory Efficiency and Large Data Processing

- **Use Generator Functions**
  - Don't load everything into memory when dealing with large datasets
  ```python
  # High memory consumption
  def get_all_data():
      return [process(x) for x in range(10000000)]
      
  # Memory efficient
  def get_data_generator():
      for x in range(10000000):
          yield process(x)
  ```

- **Process with Appropriate Chunk Sizes**
  - Don't read huge files all at once
  ```python
  # Memory efficient file reading
  def process_large_file(filename):
      with open(filename, 'r') as f:
          for chunk in iter(lambda: f.read(4096), ''):
              process_chunk(chunk)
  ```

- **Avoid Unnecessary Copies**
  - Use views or references instead of copying large data structures
  - Example: Minimize use of `.copy()` in `pandas`

## 5. Utilizing External Libraries

- **Vectorization with NumPy**
  - Use NumPy array operations instead of loops
  ```python
  # Slow
  result = []
  for i in range(1000000):
      result.append(i * 2)
      
  # Fast
  import numpy as np
  result = np.arange(1000000) * 2
  ```

- **Use Pandas for Data Processing**
  - Leverage `pandas` for tabular data operations
  ```python
  import pandas as pd
  
  # Efficiently read CSV
  df = pd.read_csv('large_file.csv')
  
  # Efficient group operations
  result = df.groupby('category').agg({'value': 'sum'})
  ```

- **Use Cython**
  - Reimplement performance-critical parts in Cython
  ```python
  # example.pyx
  def fast_function(int x, int y):
      cdef int i, result = 0
      for i in range(x):
          result += i * y
      return result
  ```

- **JIT Compilation with Numba**
  - Use Numba's JIT compilation for functions with numerical calculations
  ```python
  from numba import jit
  
  @jit(nopython=True)
  def fast_numerical_function(x, y):
      result = 0
      for i in range(len(x)):
          result += x[i] * y[i]
      return result
  ```

## 6. Profiling and Performance Measurement

- **Perform Code Profiling**
  - Use `cProfile` or `line_profiler` to identify slow parts
  ```python
  import cProfile
  cProfile.run('my_function()')
  ```

- **Measure with the `timeit` Module**
  - Actually measure the effect before and after optimization
  ```python
  import timeit
  
  # Speed comparison between original and optimized code
  original_time = timeit.timeit('original_function()', number=1000, globals=globals())
  optimized_time = timeit.timeit('optimized_function()', number=1000, globals=globals())
  print(f"Speed improvement: {original_time / optimized_time:.2f}x")
  ```

- **Optimize Based on Measurements, Not Guesses**
  - "Premature optimization is the root of all evil" - The effect may be less than expected

## 7. Parallel and Asynchronous Processing

- **Parallel Processing for CPU-bound Tasks**
  - Use `multiprocessing` to leverage multiple cores
  ```python
  from multiprocessing import Pool
  
  def process_data(data_chunk):
      # Some computational processing
      return result
      
  with Pool(processes=4) as pool:
      results = pool.map(process_data, data_chunks)
  ```

- **Asynchronous Processing for I/O-bound Tasks**
  - Consider `asyncio` or `threading` for network communication or file I/O
  ```python
  import asyncio
  import aiohttp
  
  async def fetch_url(url):
      async with aiohttp.ClientSession() as session:
          async with session.get(url) as response:
              return await response.text()
              
  async def main():
      urls = ["http://example.com", "http://python.org", "http://github.com"]
      tasks = [fetch_url(url) for url in urls]
      results = await asyncio.gather(*tasks)
  ```

- **Consider the Impact of GIL (Global Interpreter Lock)**
  - Note that with CPU-bound code, threads may not parallelize effectively due to the GIL

## 8. Advanced Optimization Techniques

- **Implement Memoization (Caching)**
  - Cache results of functions called multiple times with the same arguments
  ```python
  from functools import lru_cache
  
  @lru_cache(maxsize=128)
  def expensive_function(a, b):
      # Computationally expensive processing
      return result
  ```

- **Cache Results of Disk I/O or Network I/O**
  - Consider caching strategies for reusable data

- **Use PyPy**
  - Running with the PyPy interpreter instead of CPython can provide 2-5x speedup
  - Note: Compatibility issues may arise with code dependent on C extensions

## 9. Additional Considerations

- **Use Optimized Libraries**
  - Actively use standard library or optimized third-party libraries
  - Examples: `NumPy`/`SciPy` for numerical calculations, `pandas` for data processing

- **Consider Switching Implementation Languages**
  - For parts requiring extremely fast processing, implement in Rust, C++, or C and call from Python
  ```python
  # Example of calling C functions using ctypes
  import ctypes
  
  # Load C library
  lib = ctypes.CDLL('./libfast.so')
  
  # Set function prototype
  lib.fast_calculation.argtypes = [ctypes.c_int, ctypes.c_int]
  lib.fast_calculation.restype = ctypes.c_int
  
  # Call C function
  result = lib.fast_calculation(10, 20)
  ```

- **Verify Optimization Effects through Actual Measurement**
  - Measure how much performance improvement your optimization efforts actually bring
  - Decide based on comparing with trade-offs (code complexity, maintainability)
