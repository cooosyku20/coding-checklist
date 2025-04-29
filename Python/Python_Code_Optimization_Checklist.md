# Python Code Optimization Checklist

This checklist provides systematic optimization techniques to improve the processing and execution speed of Python code. You can incorporate it into code reviews and development processes while considering the balance between code quality and performance.

## 1. Algorithm and Data Structure Optimization

- **Computational Complexity Optimization**
  - Consider changing to more efficient algorithms (e.g., O(N²) → O(N log N) or O(N))
  - Avoid unnecessary repeated calculations

- **Optimal Data Structure Selection**
  - Use dictionaries (dict) or sets (set) instead of lists for frequent lookups
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
  - Utilize new data structures available in Python 3.9+ (e.g., new `dict` methods)
  ```python
  # Dictionary merging in Python 3.9+
  dict1 = {"a": 1, "b": 2}
  dict2 = {"c": 3, "d": 4}
  
  # Using new operator
  combined = dict1 | dict2  # {'a': 1, 'b': 2, 'c': 3, 'd': 4}
  
  # For updates
  dict1 |= dict2  # dict1 becomes {'a': 1, 'b': 2, 'c': 3, 'd': 4}
  ```

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
  - Move operations that don't need to be executed inside loops to outside the loop
  ```python
  # Slow
  for i in range(1000):
      expensive_setup()  # same setup executed each time
      process(i)
  
  # Fast
  expensive_setup()  # executed only once
  for i in range(1000):
      process(i)
  ```

  - Optimize conditional evaluations (evaluate frequent conditions first)
  ```python
  # Slow (rare condition evaluated first)
  for item in large_list:
      if rare_condition(item) and common_condition(item):
          process(item)
          
  # Fast (frequent condition evaluated first)
  for item in large_list:
      if common_condition(item) and rare_condition(item):
          process(item)
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

- **Utilize Generator Expressions**
  - Improves memory efficiency when handling large data
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
  - Minimize object creation inside loops
  - Utilize object pooling/caching
  ```python
  # Slow - new tuple created each time
  for i in range(1000000):
      coordinates = (x + i, y + i)
      process_point(coordinates)
      
  # Fast - reuse existing object
  coordinates = [0, 0]  # use list to make it mutable
  for i in range(1000000):
      coordinates[0] = x + i
      coordinates[1] = y + i
      process_point(coordinates)
  ```

## 3. Utilizing Built-in Functions and Standard Library

- **Leverage Built-in Functions**
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

- **Utilize Special Containers from the `collections` Module**
  - Properly use `defaultdict`, `Counter`, `namedtuple`, etc.
  ```python
  from collections import Counter
  
  # Count occurrences of elements
  word_counts = Counter(['apple', 'banana', 'apple', 'orange'])
  # Counter({'apple': 2, 'banana': 1, 'orange': 1})
  ```

- **Consider Function Call Overhead**
  - Bundle processing instead of calling many small functions
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
  - When handling large datasets, avoid loading everything into memory
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
  - Don't load entire large files at once
  ```python
  # Memory-efficient file reading
  def process_large_file(filename):
      with open(filename, 'r') as f:
          for chunk in iter(lambda: f.read(4096), ''):
              process_chunk(chunk)
  ```

  - Combining chunk processing with multiprocessing
  ```python
  from multiprocessing import Pool
  
  def process_file_in_parallel(filename, chunk_size=1000000, processes=4):
      # Process in chunks in parallel
      def get_chunk_positions():
          positions = [0]
          with open(filename, 'rb') as f:
              while True:
                  f.seek(chunk_size, 1)  # advance chunk_size bytes from current position
                  if f.read(1) == b'':  # reached end of file
                      break
                  positions.append(f.tell())
          return positions
      
      def process_chunk(start_pos):
          results = []
          with open(filename, 'r') as f:
              f.seek(start_pos)
              # Process chunk
              chunk_data = f.read(chunk_size)
              # Processing logic
              results = analyze_chunk(chunk_data)
          return results
      
      # Get chunk start positions
      chunk_positions = get_chunk_positions()
      
      # Execute parallel processing
      with Pool(processes=processes) as pool:
          all_results = pool.map(process_chunk, chunk_positions)
      
      # Combine results
      return combine_results(all_results)
  ```

- **Avoid Unnecessary Copies**
  - Avoid copying large data structures; use views or references
  - Example: Minimize use of `.copy()` in `pandas`
  ```python
  import pandas as pd
  
  # Avoid: Unnecessary copying
  df2 = df1.copy()  # complete copy of a large DataFrame
  df2['new_col'] = df2['col1'] * 2
  
  # Better: Create copies only when necessary
  # No copy needed for operations that don't modify the original DataFrame
  df2 = df1[['col1', 'col2']]  # creates a view (not a copy)
  df3 = df1.assign(new_col=df1['col1'] * 2)  # new DataFrame with added column
  ```

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

- **Data Processing with Pandas**
  - Utilize `pandas` for tabular data operations
  ```python
  import pandas as pd
  
  # Efficiently read CSV
  df = pd.read_csv('large_file.csv')
  
  # Efficient group operations
  result = df.groupby('category').agg({'value': 'sum'})
  ```

  - Efficient pandas usage
  ```python
  # Slow - creating intermediate DataFrames with chained operations
  df_filtered = df[df['value'] > 100]
  df_grouped = df_filtered.groupby('category')
  result = df_grouped.sum()
  
  # Fast - streaming processing with method chaining
  result = df[df['value'] > 100].groupby('category').sum()
  
  # For large datasets - chunk processing
  chunks = pd.read_csv('very_large_file.csv', chunksize=10000)
  result = pd.DataFrame()
  for chunk in chunks:
      processed = chunk[chunk['value'] > 100].groupby('category').sum()
      result = pd.concat([result, processed])
  ```

- **Using Cython**
  - Reimplement performance-critical parts in Cython
  ```python
  # example.pyx
  def fast_function(int x, int y):
      cdef int i, result = 0
      for i in range(x):
          result += i * y
      return result
  ```

- **Compilation with Numba**
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

- **Code Profiling**
  - Identify slow parts with `cProfile` or `line_profiler`
  ```python
  import cProfile
  cProfile.run('my_function()')
  ```

  - More detailed profiling
  ```python
  # Line-level profiling with line_profiler
  # pip install line_profiler
  # Command line: kernprof -l script.py
  # or
  # Command line: python -m line_profiler script.py.lprof
  
  @profile  # decorator for use with line_profiler
  def function_to_profile():
      result = 0
      for i in range(1000):
          result += i * i
      return result
  ```

- **Measurement with the `timeit` Module**
  - Actually measure effects before and after optimization
  ```python
  import timeit
  
  # Speed comparison of original vs. optimized code
  original_time = timeit.timeit('original_function()', number=1000, globals=globals())
  optimized_time = timeit.timeit('optimized_function()', number=1000, globals=globals())
  print(f"Speed improvement: {original_time / optimized_time:.2f}x")
  ```

- **Optimization Based on Measurement, Not Assumptions**
  - "Premature optimization is the root of all evil" - sometimes effects may be less than expected
  - Prioritize optimization: Identify and focus on parts that consume the most time

- **Continuous Performance Monitoring**
  - Early detection of performance degradation due to feature additions/changes
  ```python
  import time
  import statistics
  import logging
  
  def benchmark(func):
      """Decorator to measure function execution time"""
      def wrapper(*args, **kwargs):
          start_time = time.time()
          result = func(*args, **kwargs)
          end_time = time.time()
          execution_time = end_time - start_time
          
          # Log execution time
          logging.info(f"{func.__name__} executed in {execution_time:.6f} seconds")
          
          # Add execution time to statistics
          if not hasattr(wrapper, 'execution_times'):
              wrapper.execution_times = []
          wrapper.execution_times.append(execution_time)
          
          # Output statistical information periodically
          if len(wrapper.execution_times) % 100 == 0:
              avg_time = statistics.mean(wrapper.execution_times[-100:])
              logging.info(f"{func.__name__} avg execution time (last 100 calls): {avg_time:.6f} seconds")
          
          return result
      return wrapper
  
  # Usage example
  @benchmark
  def process_data(data):
      # Processing logic
      pass
  ```

## 7. Parallel and Asynchronous Processing

- **Parallel Processing for CPU-Bound Tasks**
  - Use `multiprocessing` to leverage multiple cores
  ```python
  from multiprocessing import Pool
  
  def process_data(data_chunk):
      # Some computational processing
      return result
      
  with Pool(processes=4) as pool:
      results = pool.map(process_data, data_chunks)
  ```

- **Asynchronous Processing for I/O-Bound Tasks**
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
  - Be aware that threads are not effectively parallelized for CPU-bound code due to the GIL
  - Threads are effective for I/O-bound tasks or tasks that wait for resources

- **Balance Between Parallelism Complexity and Overhead**
  - Parallelization overhead may outweigh benefits for small tasks
  - Find the appropriate granularity for task division
  ```python
  import concurrent.futures
  import time
  
  def process_chunk(data):
      # Process data chunk
      time.sleep(0.1)  # simulate processing time
      return sum(data)
  
  def test_parallelism(data_size, chunk_size, max_workers):
      """Performance test with different parallelism levels"""
      data = list(range(data_size))
      chunks = [data[i:i+chunk_size] for i in range(0, data_size, chunk_size)]
      
      # Single-threaded processing (baseline)
      start = time.time()
      serial_result = sum(process_chunk(chunk) for chunk in chunks)
      serial_time = time.time() - start
      
      # Parallel processing
      start = time.time()
      with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
          parallel_result = sum(executor.map(process_chunk, chunks))
      parallel_time = time.time() - start
      
      # Compare results
      speedup = serial_time / parallel_time
      overhead = (max_workers * parallel_time) / serial_time
      
      print(f"Data size: {data_size}, Chunk size: {chunk_size}, Workers: {max_workers}")
      print(f"Serial time: {serial_time:.4f}s, Parallel time: {parallel_time:.4f}s")
      print(f"Speedup: {speedup:.2f}x, Efficiency: {100*speedup/max_workers:.1f}%")
      
      return {
          'serial_time': serial_time,
          'parallel_time': parallel_time,
          'speedup': speedup,
          'efficiency': speedup/max_workers
      }
  
  # Run tests with different settings to find optimal parallelism
  results = []
  for chunk_size in [10, 100, 1000]:
      for workers in [2, 4, 8]:
          results.append(test_parallelism(10000, chunk_size, workers))
  
  # Identify optimal configuration from results
  best_config = max(results, key=lambda x: x['speedup'])
  ```

## 8. Advanced Optimization Techniques

- **Memoization (Caching) Implementation**
  - Cache results of functions called multiple times with the same arguments
  ```python
  from functools import lru_cache
  
  @lru_cache(maxsize=128)
  def expensive_function(a, b):
      # Computationally expensive processing
      return result
  ```

- **Caching Disk I/O or Network I/O Results**
  - Consider caching strategies for reusable data
  ```python
  import json
  import os
  import time
  import hashlib
  
  class DiskCache:
      """Simple disk-based cache implementation"""
      def __init__(self, cache_dir='.cache', expiration=3600):
          self.cache_dir = cache_dir
          self.expiration = expiration
          os.makedirs(cache_dir, exist_ok=True)
      
      def _get_cache_path(self, key):
          """Generate cache file path from key"""
          key_hash = hashlib.md5(key.encode()).hexdigest()
          return os.path.join(self.cache_dir, f"{key_hash}.json")
      
      def get(self, key, default=None):
          """Retrieve data from cache"""
          cache_path = self._get_cache_path(key)
          
          if not os.path.exists(cache_path):
              return default
          
          try:
              with open(cache_path, 'r') as f:
                  cache_data = json.load(f)
              
              # Check expiration
              if time.time() - cache_data['timestamp'] > self.expiration:
                  os.remove(cache_path)
                  return default
              
              return cache_data['data']
          except Exception:
              return default
      
      def set(self, key, data):
          """Save data to cache"""
          cache_path = self._get_cache_path(key)
          
          cache_data = {
              'timestamp': time.time(),
              'data': data
          }
          
          with open(cache_path, 'w') as f:
              json.dump(cache_data, f)
      
      def clear(self):
          """Clear cache"""
          for filename in os.listdir(self.cache_dir):
              file_path = os.path.join(self.cache_dir, filename)
              if os.path.isfile(file_path):
                  os.remove(file_path)
  
  # Usage example: API response caching
  cache = DiskCache()
  
  def fetch_api_data(url):
      """Get data from API (with caching)"""
      # Check cache
      cached_data = cache.get(url)
      if cached_data:
          print("Using cached data")
          return cached_data
      
      # Actual API request
      print(f"Fetching data from {url}")
      response = requests.get(url)
      data = response.json()
      
      # Save to cache
      cache.set(url, data)
      
      return data
  ```

- **Using PyPy**
  - Running with PyPy interpreter instead of CPython can achieve 2-5x speedup
  - Note: Compatibility issues may occur with code dependent on C extensions

- **Optimization with Type Hints**
  - Use optimizing compilers that leverage type hints (Mypy, Cython, Mypyc)
  ```python
  # Version with type hints
  def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
      return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
  
  # Typed implementation with Cython (.pyx)
  def calculate_distance_cython(double x1, double y1, double x2, double y2):
      return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
  ```

## 9. Balance Between Performance and Code Quality

- **Tradeoff Between Readability and Optimization**
  - Evaluate impact on code clarity and maintainability before applying optimization
  ```python
  # Highly optimized but difficult to read
  def optimized_but_cryptic(d, k, v):
      return k in d and d[k] == v and any(i > 0 for i in d.values())
  
  # Balanced implementation
  def balanced_implementation(data, key, value):
      # Check key existence and value
      if key not in data or data[key] != value:
          return False
      
      # Check if there's a positive value
      for item_value in data.values():
          if item_value > 0:
              return True
      
      return False
  ```

- **Prioritizing Performance Optimizations**
  - Determine level of optimization based on code importance and frequency of use
  - Focus on hotspots (frequently executed important code)
  ```python
  # Performance classification comment examples
  # [CRITICAL_PATH] - Most frequently executed code where optimization is important
  # [OPTIMIZATION_NEEDED] - Areas that should be improved for performance
  # [BOTTLENECK] - Identified bottlenecks
  
  # [CRITICAL_PATH]
  def process_frame(image_data):
      # This function is called for each frame in a video stream
      # Performance is critical
      result = np.array(image_data)
      # Optimized code...
      return result
  
  # [OPTIMIZATION_OK]
  def load_config():
      # Called only once at application startup
      # Performance is not critical
      with open('config.json', 'r') as f:
          return json.load(f)
  ```

- **Documenting Code Complexity and Optimization**
  - Clearly explain the reason and method when applying complex optimizations in comments
  ```python
  def process_large_dataset(data, batch_size=1000):
      """Efficiently process large datasets
      
      Notes:
      1. Uses batch processing to reduce memory consumption
      2. Returns results as an iterator to save memory
      3. Uses NumPy vectorized operations internally to improve processing speed
      4. batch_size parameter adjusts the size of processing batches
         - Too small increases parallel processing overhead
         - Too large increases memory consumption
      
      Optimization history:
      - 2023-06-01: Changed from simple loop to parallel batch processing (5x speedup)
      - 2023-07-15: Applied NumPy vectorization (additional 2x speedup)
      """
      # Implementation...
  ```

- **Reviewable Refactoring Approach**
  - Perform large optimizations incrementally, with testing and validation at each step
  ```python
  # Step 1: Current implementation (functional but poor performance)
  def original_implementation():
      results = []
      for i in range(1000000):
          value = complex_calculation(i)
          if filter_condition(value):
              results.append(transform(value))
      return aggregate(results)
  
  # Step 2: Algorithm improvement
  def improved_algorithm():
      # Optimization through algorithm change
      return optimized_implementation()
  
  # Step 3: Data structure optimization
  def optimized_data_structures():
      # Using more efficient data structures
      return faster_implementation()
  
  # Step 4: Parallelization
  def parallelized_implementation():
      # Add parallel processing
      return parallel_implementation()
  ```

## 10. Maintaining Debuggability

- **Logging and Monitoring Optimized Code**
  - Maintain appropriate logging even in performance-critical code
  ```python
  import logging
  import time
  
  def optimized_function(data):
      """Optimized function that maintains logging"""
      start_time = time.time()
      
      # Input data validation and logging (debug level)
      logging.debug(f"Processing data: length={len(data)}, sample={data[:5]}")
      
      try:
          # Actual processing (highly optimized)
          result = fast_processing_logic(data)
          
          # Log processing results (info level)
          processing_time = time.time() - start_time
          logging.info(f"Processing completed in {processing_time:.2f}s, results: {len(result)} items")
          
          return result
      except Exception as e:
          # Detailed error logging
          logging.exception(f"Error processing data: {e}")
          raise
  ```

- **Debug Mode Implementation**
  - Switch between optimized code and more readable debug code
  ```python
  import os
  
  # Control debug mode with environment variable
  DEBUG_MODE = os.environ.get('DEBUG', '0') == '1'
  
  def calculate_statistics(data):
      if DEBUG_MODE:
          # Debug mode: step-by-step processing and logging
          logging.debug("Starting statistics calculation")
          count = len(data)
          logging.debug(f"Count: {count}")
          
          if not data:
              logging.warning("Empty data array")
              return {"count": 0, "average": 0, "sum": 0}
          
          total = sum(data)
          logging.debug(f"Sum: {total}")
          
          average = total / count
          logging.debug(f"Average: {average}")
          
          return {"count": count, "average": average, "sum": total}
      else:
          # Production mode: optimized implementation
          if not data:
              return {"count": 0, "average": 0, "sum": 0}
          count = len(data)
          total = sum(data)
          return {"count": count, "average": total / count, "sum": total}
  ```

- **Assertions for Performance Testing**
  - Include assertions to detect bugs caused by optimization
  ```python
  def optimized_sort_and_filter(data, threshold):
      """Optimized function to sort and filter data"""
      # Precondition assertions
      assert isinstance(data, list), "data must be a list"
      assert all(isinstance(x, (int, float)) for x in data), "all items must be numeric"
      
      # Original algorithm result (for testing)
      if __debug__:
          expected = sorted([x for x in data if x > threshold])
      
      # Optimized algorithm
      # (e.g., using NumPy for speedup)
      import numpy as np
      arr = np.array(data)
      mask = arr > threshold
      filtered = arr[mask]
      result = np.sort(filtered).tolist()
      
      # Result verification (debug mode only)
      if __debug__:
          assert result == expected, "Optimized implementation returned different results"
      
      return result
  ```

- **Performance Degradation Detection**
  - Detect regressions through continuous performance monitoring
  ```python
  import time
  import json
  import os
  
  class PerformanceMonitor:
      """Utility to track performance changes"""
      def __init__(self, history_file="perf_history.json"):
          self.history_file = history_file
          self.history = self._load_history()
      
      def _load_history(self):
          """Load performance history"""
          if not os.path.exists(self.history_file):
              return {}
          
          try:
              with open(self.history_file, 'r') as f:
                  return json.load(f)
          except Exception:
              return {}
      
      def _save_history(self):
          """Save performance history"""
          with open(self.history_file, 'w') as f:
              json.dump(self.history, f, indent=2)
      
      def measure(self, func_name, func, *args, **kwargs):
          """Measure function execution time and compare with history"""
          start_time = time.time()
          result = func(*args, **kwargs)
          execution_time = time.time() - start_time
          
          # Record in history
          if func_name not in self.history:
              self.history[func_name] = []
          
          self.history[func_name].append({
              'timestamp': time.time(),
              'execution_time': execution_time
          })
          
          # Compare with average of last 5 runs
          if len(self.history[func_name]) >= 6:
              recent_times = [entry['execution_time'] for entry in self.history[func_name][-6:-1]]
              avg_time = sum(recent_times) / len(recent_times)
              
              # Warning if execution time is 20% slower
              if execution_time > avg_time * 1.2:
                  logging.warning(
                      f"Performance degradation detected in {func_name}: "
                      f"Current: {execution_time:.4f}s, Avg: {avg_time:.4f}s "
                      f"({(execution_time/avg_time-1)*100:.1f}% slower)"
                  )
          
          # Save history
          self._save_history()
          
          return result
  
  # Usage example
  monitor = PerformanceMonitor()
  
  def process_data(data):
      # Processing logic
      pass
  
  # Measure performance
  result = monitor.measure("process_data", process_data, sample_data)
  ```

## 11. Optimization While Maintaining Code Extensibility

- **Separating Interface and Implementation**
  - Hide optimized code as implementation details and provide a clean interface
  ```python
  # Stable public interface
  def calculate_statistics(data):
      """Calculate statistics for data
      
      Args:
          data: List of numbers
          
      Returns:
          Dictionary of statistics (mean, median, std deviation, etc.)
      """
      return _optimized_statistics_impl(data)
  
  # Internal implementation (optimized but may change)
  def _optimized_statistics_impl(data):
      """Optimized implementation of statistics calculation
      
      Note: This is an internal API and may change.
      Don't call directly, use calculate_statistics() instead.
      """
      import numpy as np
      arr = np.array(data)
      return {
          'mean': float(np.mean(arr)),
          'median': float(np.median(arr)),
          'std': float(np.std(arr)),
          'min': float(np.min(arr)),
          'max': float(np.max(arr)),
      }
  ```

- **Balancing Extensible Design and Optimization**
  - Balance abstraction and optimization
  ```python
  from abc import ABC, abstractmethod
  
  # Abstract base class (interface)
  class DataProcessor(ABC):
      @abstractmethod
      def process(self, data):
          """Abstract method to process data"""
          pass
  
  # Optimized implementation
  class OptimizedProcessor(DataProcessor):
      def process(self, data):
          """Optimized data processing implementation"""
          # Highly optimized code
          return optimized_result
  
  # Another implementation (e.g., for debugging)
  class DebuggableProcessor(DataProcessor):
      def process(self, data):
          """Implementation with detailed logging"""
          # Easily debuggable code
          return debug_result
  
  # Factory function
  def get_processor(optimize=True, debug=False):
      """Return appropriate processor instance"""
      if debug:
          return DebuggableProcessor()
      elif optimize:
          return OptimizedProcessor()
      else:
          return StandardProcessor()
  
  # Usage example
  processor = get_processor(optimize=True)
  result = processor.process(data)
  ```

- **Optimization with Future Extensions in Mind**
  - Design to allow feature extensions while maintaining performance
  ```python
  class ImageProcessor:
      """Image processing class
      
      Extensibility-focused design:
      1. Pluggable filter system
      2. Fine-grained optimization of performance-critical parts
      3. Configurable processing pipeline
      """
      def __init__(self):
          self.filters = {}  # Registered filters
          self.pipeline = []  # Processing pipeline
      
      def register_filter(self, name, filter_func):
          """Register a new filter"""
          self.filters[name] = filter_func
      
      def set_pipeline(self, pipeline):
          """Set processing pipeline"""
          # Validate pipeline
          for step in pipeline:
              if step not in self.filters:
                  raise ValueError(f"Unknown filter: {step}")
          self.pipeline = pipeline
      
      def process(self, image):
          """Process image"""
          result = image.copy()  # Don't modify original image
          
          # Process according to pipeline
          for filter_name in self.pipeline:
              filter_func = self.filters[filter_name]
              result = filter_func(result)
          
          return result
  
  # Optimized filter implementation
  def optimized_blur(image):
      """Optimized blur filter"""
      # Fast implementation using NumPy or other libraries
      return blurred_image
  
  # Processor configuration
  processor = ImageProcessor()
  processor.register_filter("blur", optimized_blur)
  processor.register_filter("sharpen", optimized_sharpen)
  processor.set_pipeline(["blur", "sharpen"])
  
  # Execute processing
  processed = processor.process(input_image)
  ```

## 12. Domain-Specific Optimization Patterns

### Web Application Optimizations

- **Database Query Optimization**
  ```python
  # Slow - loading more data than needed
  users = db.query(User).all()
  active_users = [user for user in users if user.is_active]
  
  # Fast - filtering at database level
  active_users = db.query(User).filter(User.is_active == True).all()
  
  # Further optimized - loading only required columns
  active_user_ids = db.query(User.id).filter(User.is_active == True).all()
  ```

- **Effective Use of Caching**
  ```python
  from functools import lru_cache
  from django.core.cache import cache
  
  # View-level cache (Django example)
  @lru_cache(maxsize=128)
  def expensive_calculation(param1, param2):
      # Computationally expensive processing
      return result
  
  # Database result caching
  def get_popular_products(category_id, limit=10):
      cache_key = f"popular_products:{category_id}:{limit}"
      
      # Check cache
      cached_result = cache.get(cache_key)
      if cached_result:
          return cached_result
      
      # Get from DB
      products = db.query(Product).\
          filter(Product.category_id == category_id).\
          order_by(Product.views.desc()).\
          limit(limit).all()
      
      # Save to cache (1 hour expiration)
      cache.set(cache_key, products, timeout=3600)
      
      return products
  ```

- **Asynchronous APIs and Celery Tasks**
  ```python
  # Move time-consuming processes to asynchronous tasks (Celery example)
  from celery import shared_task
  
  @shared_task
  def process_large_report(report_id):
      # Report generation (time-consuming process)
      report = generate_report(report_id)
      # Completion notification
      send_notification(report.user_id, "Report is ready")
      
  # API endpoint
  def start_report_generation(request, report_id):
      # Start async task
      task = process_large_report.delay(report_id)
      # Return task ID
      return {"task_id": task.id, "status": "processing"}
  ```

### Data Processing Application Optimizations

- **Efficient CSV/Large File Processing**
  ```python
  import csv
  from itertools import islice
  
  def process_large_csv(file_path, batch_size=10000):
      """Efficiently process large CSV files"""
      with open(file_path, 'r', newline='') as file:
          reader = csv.DictReader(file)
          
          # Batch processing
          batch = []
          for row in reader:
              batch.append(row)
              
              # Process when batch size is reached
              if len(batch) >= batch_size:
                  process_batch(batch)
                  batch = []
          
          # Process remaining batch
          if batch:
              process_batch(batch)
  ```

- **Efficient Aggregation of Large Data**
  ```python
  import pandas as pd
  from collections import defaultdict
  
  def aggregate_large_dataset(file_path):
      """Memory-efficient large data aggregation"""
      # Dictionary for intermediate aggregation
      aggregates = defaultdict(lambda: {"count": 0, "sum": 0})
      
      # Chunk processing
      for chunk in pd.read_csv(file_path, chunksize=100000):
          # Aggregate in each chunk
          for category, group in chunk.groupby('category'):
              aggregates[category]["count"] += len(group)
              aggregates[category]["sum"] += group['value'].sum()
      
      # Calculate final results
      results = {
          category: {
              "count": data["count"],
              "sum": data["sum"],
              "average": data["sum"] / data["count"] if data["count"] > 0 else 0
          }
          for category, data in aggregates.items()
      }
      
      return results
  ```

### Machine Learning Application Optimizations

- **Data Preprocessing Optimization**
  ```python
  import numpy as np
  from sklearn.preprocessing import StandardScaler
  
  def optimize_preprocessing(X, batch_size=1000):
      """Optimize preprocessing of large datasets"""
      n_samples = X.shape[0]
      
      # Indices for batch processing
      indices = np.arange(n_samples)
      
      # Train scaler on sampled data
      sample_indices = np.random.choice(indices, min(10000, n_samples), replace=False)
      scaler = StandardScaler().fit(X[sample_indices])
      
      # Transform in batches
      X_transformed = np.zeros_like(X)
      for start_idx in range(0, n_samples, batch_size):
          end_idx = min(start_idx + batch_size, n_samples)
          batch_indices = indices[start_idx:end_idx]
          X_transformed[batch_indices] = scaler.transform(X[batch_indices])
      
      return X_transformed
  ```

- **Model Inference Optimization**
  ```python
  import numpy as np
  import onnxruntime
  
  class OptimizedPredictor:
      """Optimized model inference class"""
      def __init__(self, model_path, batch_size=32):
          # Initialize ONNX runtime session
          self.session = onnxruntime.InferenceSession(model_path)
          self.input_name = self.session.get_inputs()[0].name
          self.batch_size = batch_size
      
      def predict(self, data):
          """Efficient inference with batch processing"""
          n_samples = len(data)
          results = []
          
          # Batch processing
          for start_idx in range(0, n_samples, self.batch_size):
              end_idx = min(start_idx + self.batch_size, n_samples)
              batch = data[start_idx:end_idx]
              
              # Execute inference
              outputs = self.session.run(None, {self.input_name: batch})
              results.append(outputs[0])
          
          # Combine results
          return np.vstack(results)
  ```

## 13. Other Considerations

- **Utilize Optimized Libraries**
  - Actively use standard library and optimized third-party libraries
  - Examples: `NumPy`/`SciPy` for numerical computation, `pandas` for data processing

- **Consider Switching Implementation Language**
  - Implement extremely performance-critical parts in Rust, C++, or C and call from Python
  ```python
  # Example of calling C functions with ctypes
  import ctypes
  
  # Load C library
  lib = ctypes.CDLL('./libfast.so')
  
  # Set function prototype
  lib.fast_calculation.argtypes = [ctypes.c_int, ctypes.c_int]
  lib.fast_calculation.restype = ctypes.c_int
  
  # Call C function
  result = lib.fast_calculation(10, 20)
  ```

- **Measure to Confirm Optimization Effects**
  - Measure how much performance improvement your optimization efforts actually bring
  - Compare with tradeoffs (code complexity, maintainability)

- **Optimization Decision Criteria**
  ```
  Optimization decision checklist:
  
  1. Is this code executed frequently?
     - Is it in the application's critical path
     - Does it directly affect user experience
     - Is it called with high frequency even in background processing
  
  2. Is current performance problematic?
     - Is it an actually measured bottleneck
     - Does it fail to meet user expectations or requirements
     - Are there concerns about future scalability
  
  3. Do optimization benefits outweigh costs?
     - Decreased maintainability due to code complexity
     - Increased debugging difficulty
     - Increased development time
  
  4. Are there alternatives to optimization?
     - Algorithm changes
     - Caching introduction
     - Distributed processing
     - Hardware scaling
  
  5. Are there side effects from optimization?
     - Impact on other processes
     - Increased memory usage
     - Increased overall system complexity
  ```

- **Continuous Performance Monitoring and Maintenance**
  - Automated performance testing
  - Regular profiling
  - Maintaining performance-related documentation
