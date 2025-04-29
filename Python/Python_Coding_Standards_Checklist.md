# Python Coding Standards Checklist

## Legend

**Application Context**:
- [General]: All Python code
- [Script]: Single script files
- [Library]: Libraries/packages
- [App]: Large-scale applications

**Priority**:
- [Required]: Essential for basic code quality
- [Important]: Greatly improves code maintainability and readability
- [Recommended]: Best practices for better code

**Category**:
- [Code]: Items related to writing code
- [Design]: Items related to code design and structure
- [Tools]: Items related to tools

## 1. Pythonic Code Style [Code]

- [Required] Follow important PEP 8 style rules [General]
    - Use 4 spaces for indentation
    - Maximum line length of 79 characters (72 for docstrings and comments) *Note: Follow team-agreed length like 88 characters if using `black`*
    - Use blank lines: 2 between functions/classes, 1 between methods
    - Write imports on separate lines, ordered by standard library, third-party, local application *Note: Recommend automatic sorting with `isort`*

- [Required] Adhere to Python naming conventions [General]
    - Class names: CapWords (e.g., `MyClass`)
    - Function/variable/method names: snake_case (e.g., `my_function`)
    - Constants: UPPER_SNAKE_CASE (e.g., `MAX_VALUE`)
    - Private attributes: Leading underscore (e.g., `_private_var`)

- [Important] Use type hints appropriately [General] (Python 3.5+)
    - Add type hints for function parameters and return values
    ```python
    from typing import Optional

    # Recommended for Python 3.9+
    def get_user_by_id(user_id: int) -> Optional[User]:
        ...

    # Dataclass example for Python 3.7+
    from dataclasses import dataclass

    @dataclass
    class User:
        id: int
        name: str
    ```
    - For Python 3.9+, use built-in generic types like `list[int]`, `dict[str, int]`

- [Important] Use f-strings (Python 3.6+) [General]
    - Prefer f-strings over `%` or `.format()` for complex string formatting
    ```python
    # Good example
    name = "Alice"
    age = 30
    message = f"{name} is {age} years old"

    # Examples to avoid
    message = "%s is %d years old" % (name, age)
    message = "{} is {} years old".format(name, age)
    ```

- [Important] Appropriately use list comprehensions, generator expressions, and dictionary comprehensions [General]
    ```python
    # Good example: List comprehension
    squares = [x*x for x in range(10) if x % 2 == 0]

    # Good example: Generator expression (memory efficient)
    sum_squares = sum(x*x for x in range(1000))
    ```

- [Recommended] Use context managers (with statements) to ensure proper resource management [General]
    ```python
    # Good example
    try:
        with open("file.txt", "r") as f:
            content = f.read()
    except FileNotFoundError:
        # Error handling
        pass

    # Example to avoid
    f = open("file.txt", "r") # Possible FileNotFoundError
    try:
        content = f.read()
    finally:
        f.close()  # Will execute on exception, but verbose
    ```

## 2. Code Structure and Design [Design]

- [Required] Apply the Single Responsibility Principle [General]
    - Functions and methods should perform only one task
    - Classes should represent a single concept or responsibility
    - Aim to keep functions and methods under 50 lines *Note: Also consider cyclomatic complexity*

- [Required] Organize code into modules and packages [Library] [App]
    - Structure code into logical modules grouped by functionality
    - Provide clear `__init__.py` files that expose public API
    - Implement internal modules with leading underscores (e.g., `_internal.py`)

- [Important] Follow SOLID principles for object-oriented design [Library] [App]
    - Single Responsibility Principle: Each class has one responsibility
    - Open/Closed Principle: Open for extension, closed for modification
    - Liskov Substitution Principle: Derived classes can substitute base classes
    - Interface Segregation Principle: Many specific interfaces are better than one general interface
    - Dependency Inversion Principle: Depend on abstractions, not concrete implementations

- [Important] Use dependency injection to reduce coupling [Library] [App]
    - Pass dependencies as arguments rather than creating them inside functions/classes
    ```python
    # Good example
    def process_data(data, logger):
        logger.info("Processing data")
        # Process data...

    # Example to avoid
    def process_data(data):
        logger = Logger()  # Hard-coded dependency
        logger.info("Processing data")
        # Process data...
    ```

- [Important] Follow the principle of least surprise [General]
    - Functions and classes should behave as their names suggest
    - APIs should be intuitive and consistent
    - Avoid hidden side effects in functions

- [Recommended] Prefer composition over inheritance [General]
    - Use inheritance only when a true "is-a" relationship exists
    - Favor composition ("has-a") for most code reuse scenarios
    ```python
    # Composition example
    class Engine:
        def start(self):
            return "Engine started"

    class Car:
        def __init__(self, engine):
            self.engine = engine  # Composition

        def start(self):
            return f"Car starting: {self.engine.start()}"
    ```

## 3. Documentation and Comments [Code]

- [Required] Write docstrings for all public modules, functions, classes, and methods [Library]
    - Follow Google, NumPy, or reStructuredText format consistently
    - Include parameter descriptions, return values, raises exceptions, and examples
    ```python
    def calculate_average(numbers):
        """Calculate the average of a list of numbers.

        Args:
            numbers (list of int or float): The numbers to average.

        Returns:
            float: The average value.

        Raises:
            ValueError: If the input list is empty.
            TypeError: If the input contains non-numeric values.

        Examples:
            >>> calculate_average([1, 2, 3, 4])
            2.5
        """
        if not numbers:
            raise ValueError("Cannot calculate average of empty list")
        
        return sum(numbers) / len(numbers)
    ```

- [Important] Create a README file with usage examples [Library]
    - Include installation instructions, basic usage, and common scenarios
    - Provide links to detailed documentation if available

- [Important] Add comments to explain "why", not "what" [General]
    - Comment complex algorithms or unusual code
    - Explain the reasoning behind non-obvious design decisions
    ```python
    # Good comment
    # Using bisect here for O(log n) search instead of linear search
    import bisect
    bisect.insort(sorted_list, new_element)

    # Unnecessary comment
    # Increment counter
    counter += 1
    ```

- [Recommended] Document API compatibility and version constraints [Library]
    - Use explicit version ranges in `setup.py` or `pyproject.toml`
    - Clearly mark deprecated features and provide migration paths

## 4. Error Handling and Logging [Code]

- [Required] Use specific exceptions and handle them properly [General]
    - Catch specific exceptions rather than using bare `except:`
    - Define custom exceptions for your application's error cases
    ```python
    # Good example
    try:
        value = data[key]
    except KeyError:
        # Handle missing key specifically
        value = default_value

    # Example to avoid
    try:
        value = data[key]
    except:  # Catches ALL exceptions, including KeyboardInterrupt, SystemExit
        value = default_value
    ```

- [Required] Validate inputs to functions and methods [General]
    - Check parameters at the beginning of functions
    - Provide clear error messages that explain what went wrong and how to fix it
    ```python
    def process_user_data(user_id, data):
        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError(f"user_id must be a positive integer, got {user_id}")
        
        if not data:
            raise ValueError("data cannot be empty")
        
        # Process validated data...
    ```

- [Important] Use a proper logging framework [App]
    - Use the `logging` module instead of `print` statements
    - Configure appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Include contextual information in log messages (timestamps, module, function)
    ```python
    import logging

    # Configure once at application startup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    def process_order(order_id):
        logger.info(f"Processing order {order_id}")
        try:
            # Process order...
            logger.debug(f"Order {order_id} details: {order_details}")
        except Exception as e:
            logger.error(f"Failed to process order {order_id}: {e}", exc_info=True)
            raise
    ```

- [Important] Fail fast and provide clean error recovery [General]
    - Detect errors as early as possible
    - Clean up resources in `finally` blocks or context managers
    - Consider the use of `try`/`except`/`else`/`finally` structure

- [Recommended] Use defensive programming but don't overdo it [General]
    - Add assertions for internal invariants and preconditions (`assert` statements)
    - Balance between robustness and unnecessary complexity

## 5. Testing [Code] [Tools]

- [Required] Write unit tests for all public functions and methods [Library] [App]
    - Use `pytest` or `unittest` frameworks
    - Aim for high test coverage, especially for critical paths
    ```python
    # Using pytest
    def test_calculate_average():
        assert calculate_average([1, 2, 3, 4]) == 2.5
        
        with pytest.raises(ValueError):
            calculate_average([])
            
        with pytest.raises(TypeError):
            calculate_average([1, 'two', 3])
    ```

- [Important] Use fixtures and parametrized tests to improve test maintainability [Library] [App]
    - Create reusable fixtures for complex test setups
    - Use parametrization to test multiple inputs without code duplication
    ```python
    # Pytest fixture example
    @pytest.fixture
    def sample_user():
        return User(id=1, name="Test User")
        
    @pytest.mark.parametrize("numbers,expected", [
        ([1, 2, 3, 4], 2.5),
        ([0, 0, 0], 0),
        ([5], 5),
    ])
    def test_calculate_average_parametrized(numbers, expected):
        assert calculate_average(numbers) == expected
    ```

- [Important] Write integration tests for complex components [App]
    - Test interactions between modules
    - Consider database, network, and file system operations
    - Use dependency injection to enable easier testing with mocks

- [Recommended] Use property-based testing for suitable functions [Library]
    - Consider tools like Hypothesis to find edge cases automatically
    ```python
    from hypothesis import given
    from hypothesis import strategies as st

    @given(st.lists(st.integers(), min_size=1))
    def test_average_properties(numbers):
        avg = calculate_average(numbers)
        assert min(numbers) <= avg <= max(numbers)
    ```

- [Recommended] Set up continuous integration with automated testing [Library] [App]
    - Configure GitHub Actions, Travis CI, or similar tools
    - Run tests on multiple Python versions if needed
    - Include linting and type checking in CI pipeline

## 6. Package Management and Project Structure [Tools]

- [Required] Use a proper dependency management system [Library] [App]
    - Specify dependencies with version constraints in `pyproject.toml`, `setup.py`, or `requirements.txt`
    - Consider using virtual environments for development
    - Document Python version requirements

- [Important] Follow standard project structure conventions [Library]
    - Create a clear and standard directory structure
    - Use `src` layout for packages to be installed
    ```
    mypackage/
    ├── pyproject.toml
    ├── README.md
    ├── src/
    │   └── mypackage/
    │       ├── __init__.py
    │       ├── module1.py
    │       └── module2.py
    └── tests/
        ├── __init__.py
        ├── test_module1.py
        └── test_module2.py
    ```

- [Important] Use static type checking [Library] [App]
    - Run mypy to verify type annotations
    - Include mypy configuration in project setup
    - Be consistent with where/how you use type annotations

- [Recommended] Set up automated code formatting [Library] [App]
    - Use tools like `black` for consistent formatting
    - Configure `isort` for import organization
    - Automate with pre-commit hooks or CI

## 7. Performance Considerations [Code]

- [Important] Choose appropriate data structures for operations [General]
    - Use sets for membership testing (O(1)) instead of lists (O(n))
    - Use dictionaries for lookups by key
    - Consider specialized structures from `collections` module
    ```python
    # Use set for efficient membership testing
    valid_users = {user.id for user in all_users}
    if user_id in valid_users:  # O(1) operation
        process_user(user_id)
    
    # Use defaultdict to simplify code
    from collections import defaultdict
    
    counts = defaultdict(int)
    for item in items:
        counts[item] += 1  # No need to check if key exists
    ```

- [Important] Be aware of Python's performance characteristics [General]
    - Understand common bottlenecks like global variable access
    - Consider generators for processing large datasets
    - Know how closures and function calls affect performance

- [Recommended] Profile code before optimizing [General]
    - Use `cProfile`, `line_profiler`, or other profiling tools
    - Identify actual bottlenecks rather than guessing
    - Follow the principle of "premature optimization is the root of all evil"

- [Recommended] Consider NumPy, Pandas, etc. for numerical operations [General]
    - Use specialized libraries for heavy numerical computations
    - Vectorize operations when possible
    ```python
    # Using NumPy for efficient operations
    import numpy as np
    
    # Much faster than Python's built-in lists for large datasets
    data = np.array([1, 2, 3, 4, 5])
    result = data * 2  # Vectorized operation
    ```

## 8. Security Considerations [Code]

- [Required] Validate and sanitize all user inputs [App]
    - Never trust user input without validation
    - Use proper escaping when constructing SQL, HTML, etc.
    - Consider input length limits and content restrictions

- [Required] Use secure coding practices for sensitive data [App]
    - Never hardcode secrets (passwords, API keys, etc.)
    - Use environment variables or secure vaults for secrets
    - Hash passwords using proper algorithms (e.g., bcrypt)

- [Important] Handle sensitive data properly [App]
    - Minimize storing sensitive data
    - Clear sensitive data from memory when no longer needed
    - Be careful about logging sensitive information

- [Important] Implement proper authentication and authorization [App]
    - Use established libraries for authentication
    - Implement principle of least privilege
    - Properly secure API endpoints and resources

- [Recommended] Stay informed about security best practices [General]
    - Keep dependencies up to date
    - Use security scanning tools for your codebase
    - Review OWASP guidelines for web applications

## 9. Code Review and Collaboration [Process]

- [Required] Have all code reviewed before merging [Team]
    - Establish clear review criteria
    - Use pull/merge requests
    - Document the review process

- [Important] Provide helpful code review comments [Team]
    - Be specific about what needs changing and why
    - Suggest alternatives when possible
    - Focus on the code, not the person

- [Important] Establish coding standards for the team [Team]
    - Agree on formatting, naming, and design conventions
    - Document exceptions to general standards
    - Use automated tools to enforce standards where possible

- [Recommended] Use pair programming for complex features [Team]
    - Collaborate on difficult problems
    - Share knowledge across the team
    - Reduce defects through real-time review

## 10. Maintainability Best Practices [Code] [Design]

- [Required] Keep code DRY (Don't Repeat Yourself) [General]
    - Extract repeated code into functions or classes
    - Use inheritance or composition appropriately
    - Consider templating or code generation for highly similar structures

- [Important] Write self-documenting code [General]
    - Choose clear, descriptive names
    - Make code structure reveal intention
    - Use well-named helper functions for complex operations

- [Important] Follow the YAGNI principle (You Aren't Gonna Need It) [General]
    - Only implement features you need now
    - Avoid speculative generalization
    - Refactor when requirements become clearer

- [Recommended] Document technical debt [General]
    - Use `# TODO:` or similar markers for future improvements
    - Consider tracking technical debt items in your issue tracker
    - Plan for periodic refactoring to address debt

- [Recommended] Keep functions and methods small and focused [General]
    - Aim for functions that do one thing well
    - Extract complex logic into helper functions
    - Consider readability over excessive granularity
