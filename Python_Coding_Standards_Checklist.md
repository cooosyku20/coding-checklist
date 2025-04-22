# Python Coding Standards Checklist

## Legend

**Application Context**:
- [General]: All Python code
- [Script]: Single script file
- [Library]: Library/package
- [App]: Large application

**Priority**:
- [Required]: Essential for basic code quality
- [Important]: Significantly improves code maintainability and readability
- [Recommended]: Best practices for better code

**Category**:
- [Code]: Items related to coding style
- [Design]: Items related to code design/structure
- [Tool]: Items related to tools

## 1. Pythonic Coding Style [Code]

- [Required] Follow important PEP 8 style rules [General]
    - Use 4 spaces for indentation
    - Maximum line length of 79 characters (72 for docstrings and comments) *Note: If using `black`, follow the team-agreed line length, such as the default 88 characters*
    - Use blank lines: 2 between functions/classes, 1 between methods
    - Write imports on separate lines, grouped in the order of standard library, third-party, local application *Note: Automatic sorting with `isort` is recommended*

- [Required] Follow Python naming conventions [General]
    - Class names: CapWords (e.g., `MyClass`)
    - Function/variable/method names: snake_case (e.g., `my_function`)
    - Constants: UPPER_SNAKE_CASE (e.g., `MAX_VALUE`)
    - Private attributes: leading underscore (e.g., `_private_var`)

- [Important] Use type hints appropriately [General] (Python 3.5+)
    - Add type hints to function parameters and return values
    ```python
    from typing import Optional

    # Recommended for Python 3.9+
    def get_user_by_id(user_id: int) -> Optional[User]:
        ...

    # Python 3.7+ dataclass example
    from dataclasses import dataclass

    @dataclass
    class User:
        id: int
        name: str
    ```
    - Use built-in generics like `list[int]`, `dict[str, int]` for Python 3.9+

- [Important] Use f-strings (Python 3.6+) [General]
    - Use f-strings instead of `%` or `.format()` for complex string formatting
    ```python
    # Good example
    name = "Alice"
    age = 30
    message = f"{name} is {age} years old"

    # Examples to avoid
    message = "%s is %d years old" % (name, age)
    message = "{} is {} years old".format(name, age)
    ```

- [Important] Use list comprehensions, generator expressions, and dictionary comprehensions appropriately [General]
    ```python
    # Good example: List comprehension
    squares = [x*x for x in range(10) if x % 2 == 0]

    # Good example: Generator expression (memory efficient)
    sum_squares = sum(x*x for x in range(1000))
    ```

- [Recommended] Use context managers (with statements) for reliable resource management [General]
    ```python
    # Good example
    try:
        with open("file.txt", "r") as f:
            content = f.read()
    except FileNotFoundError:
        # Error handling
        pass

    # Example to avoid
    f = open("file.txt", "r") # Potential FileNotFoundError
    try:
        content = f.read()
    finally:
        f.close()  # Guaranteed to run even with exceptions, but verbose
    ```

## 2. Code Structure and Design [Design]

- [Required] Apply the Single Responsibility Principle [General]
    - Functions and methods should perform only one task
    - Classes should represent a single concept or responsibility
    - Keep functions and methods under 50 lines as a guideline *Note: Also consider cyclomatic complexity*

- [Important] Practice Separation of Concerns [Library] [App]
    - Separate data access, business logic, and presentation
    - Example: Separate database operations, data processing, and API response generation into different functions or classes
    ```python
    # Good example
    def get_user_from_db(user_id: int) -> User: ...      # Data access
    def calculate_user_stats(user: User) -> Stats: ...    # Business logic
    def format_user_response(user: User, stats: Stats) -> dict: ... # Presentation
    ```

- [Important] Use abstract classes or protocols to separate implementation [Library] [App]
    ```python
    # Example using protocols in Python 3.8+
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class DataStore(Protocol):
        def save(self, data: dict) -> bool: ...
        def load(self, id: str) -> dict: ...

    # Implementation classes just need to conform to the Protocol
    class MongoDBStore: # Implementation class doesn't need to inherit from DataStore
        def save(self, data: dict) -> bool: ...
        def load(self, id: str) -> dict: ...

    def process_data(store: DataStore, data_id: str):
        if isinstance(store, DataStore): # Possible due to runtime_checkable
             data = store.load(data_id)
             # ... process data ...
             store.save(data)
    ```

- [Recommended] Prefer composition over inheritance [Library] [App]
    ```python
    # Example to avoid (inheritance) - when not a genuine "is-a" relationship
    class SpecialDict(dict):
        def special_method(self): ...

    # Good example (composition) - "has-a" relationship
    class SpecialContainer:
        def __init__(self):
            self._data = {}

        def __getitem__(self, key): # Provide dictionary-like access if needed
            return self._data[key]

        def __setitem__(self, key, value):
            self._data[key] = value

        def special_method(self): ...
    ```

- [Recommended] Separate main processing into functions even in scripts [Script]
    ```python
    def main():
        # Main processing goes here
        print("Executing main function...")
        pass

    if __name__ == "__main__":
        main()
    ```

## 3. Error Handling and Logging [Code]

- [Required] Catch exceptions with specific types [General]
    ```python
    import logging
    logger = logging.getLogger(__name__)

    # Good example
    try:
        # ... some operation ...
        result = 1 / 0
    except FileNotFoundError:
        logger.error("Configuration file not found.")
    except ZeroDivisionError as e:
        logger.error(f"Division error occurred: {e}")
    except Exception as e: # For unexpected errors, but be as specific as possible
        logger.exception(f"An unexpected error occurred: {e}") # Logs stack trace too

    # Example to avoid
    try:
        # ... some operation ...
        pass
    except Exception:  # Difficult to determine what happened
        logger.error("An error occurred")
    ```

- [Important] Use `raise ... from ...` to maintain the exception chain [General]
    ```python
    class ProcessingError(Exception):
        pass

    def process_data():
        try:
            # ... data processing ...
            value = int("abc")
        except ValueError as e:
            # Raise a new exception with the original one (ValueError) as the cause
            raise ProcessingError("Data processing failed due to invalid format") from e

    try:
        process_data()
    except ProcessingError as e:
        logger.error(f"Processing error: {e}")
        if e.__cause__:
            logger.error(f"  Caused by: {e.__cause__}")
    ```

- [Important] Use custom exception classes to categorize exceptions [Library] [App]
    ```python
    # Application-specific base exception
    class AppError(Exception):
        """Common base exception for the application"""

    # Specific category exceptions
    class ValidationError(AppError):
        """Error related to input validation"""

    class ResourceNotFoundError(AppError):
        """Error when a resource is not found"""

    def get_resource(resource_id: str):
        if not validate_id(resource_id):
             raise ValidationError("Invalid resource ID format")
        resource = db.find(resource_id)
        if resource is None:
             raise ResourceNotFoundError(f"Resource {resource_id} not found")
        return resource
    ```

- [Important] Use log levels appropriately [App]
    - DEBUG: Detailed information, useful only for debugging
    - INFO: Confirmation of normal operations (e.g., server startup, request received)
    - WARNING: Potential issue indicators (e.g., missing configuration, deprecated API usage)
    - ERROR: Error events (program can continue running but some functionality failed) (e.g., specific request processing failure)
    - CRITICAL: Critical errors (potential program termination) (e.g., database connection failure, startup failure)

- [Important] Record structured log messages [App]
    ```python
    # Good example (standard logging + extra)
    logger.error(
        "Failed to process user data",
        extra={"user_id": 123, "action": "update", "error_code": 500}
    )

    # Or use a structured logging library (e.g., structlog)
    # structured_logger.error("Failed to process user data", user_id=123, action="update", error_code=500)
    ```

- [Recommended] Avoid empty except blocks and include a comment explaining when using pass [General]
    ```python
    # Good example
    try:
        # Remove if exists, do nothing if not
        os.remove("temp_file.tmp")
    except FileNotFoundError:
        # Ignoring because file doesn't need to be removed if it doesn't exist
        pass
    except Exception as e:
        logger.warning(f"Could not remove temp file: {e}")

    # Example to avoid
    try:
        # ... some operation ...
    except: # Dangerous to catch all exceptions silently
        pass
    ```

## 4. Data and State Management [Code]

- [Required] Minimize the use of global and class variables [General]
    - Pass state as function/method arguments or manage as instance variables in classes
    - Keep scope as narrow as possible

- [Important] Prefer immutable data structures [General]
    - Use tuples instead of lists for sequences that don't need to change
    - Use named tuples (`collections.namedtuple`) or dataclasses (`dataclasses.dataclass`, especially with `frozen=True`) instead of dictionaries for simple data structures
    ```python
    from collections import namedtuple
    from dataclasses import dataclass

    # namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    p1 = Point(10, 20)
    # p1.x = 15 # AttributeError: can't set attribute

    # dataclass (immutable)
    @dataclass(frozen=True)
    class UserConfig:
        theme: str
        notifications_enabled: bool

    config = UserConfig("dark", True)
    # config.theme = "light" # dataclasses.FrozenInstanceError
    ```

- [Important] Use Dependency Injection pattern to manage external dependencies [Library] [App]
    ```python
    # Dependency (interface/implementation)
    class DatabaseClient(Protocol):
        def get_user(self, user_id: int) -> dict: ...

    class RealDatabaseClient:
        def get_user(self, user_id: int) -> dict: ... # Actual DB access

    class MockDatabaseClient:
        def get_user(self, user_id: int) -> dict: return {"id": user_id, "name": "Mock User"}

    # Dependent (receives dependency via constructor)
    class UserService:
        def __init__(self, db_client: DatabaseClient):
            self._db = db_client

        def get_user_name(self, user_id: int) -> str:
            user_data = self._db.get_user(user_id)
            return user_data.get("name", "Unknown")

    # Usage example (production)
    real_db = RealDatabaseClient()
    user_service = UserService(real_db)
    print(user_service.get_user_name(1))

    # Usage example (testing)
    mock_db = MockDatabaseClient()
    test_user_service = UserService(mock_db)
    assert test_user_service.get_user_name(1) == "Mock User"
    ```

- [Recommended] Use module-level instances for singletons if needed [Library] [App]
    ```python
    # myapp/config.py
    import os

    class AppConfig:
        def __init__(self):
            # Load configuration (e.g., from environment variables)
            self.db_url = os.environ.get("DATABASE_URL", "sqlite:///:memory:")
            self.log_level = os.environ.get("LOG_LEVEL", "INFO")
            # ... other settings ...

    # Create instance when module is loaded
    _config_instance = AppConfig()

    def get_config() -> AppConfig:
        """Get the singleton instance of the application configuration"""
        return _config_instance

    # Usage example
    # from myapp.config import get_config
    # config = get_config()
    # print(config.db_url)
    ```
    *Note: Consider drawbacks like global state and testability. Dependency injection is often preferable.*

- [Recommended] Use class methods (`@classmethod`) and static methods (`@staticmethod`) appropriately [General]
    ```python
    class User:
        min_age = 18 # Class variable

        def __init__(self, name: str, age: int):
            if not self.validate_age(age): # Using static method
                 raise ValueError(f"Age must be at least {self.min_age}")
            self.name = name
            self.age = age

        # Instance method - accesses instance state (self)
        def greet(self) -> str:
            return f"Hello, I am {self.name}."

        # Class method - accesses class state (cls), used for alternative constructors etc.
        @classmethod
        def create_guest(cls) -> 'User':
             # References class variable min_age
            return cls(name="Guest", age=cls.min_age)

        # Static method - requires neither instance (self) nor class (cls)
        @staticmethod
        def validate_age(age: int) -> bool:
             # Cannot access class variable min_age (no self or cls)
             # Logic that only depends on constants, global variables, or arguments
            return isinstance(age, int) and age >= 18 # Hardcoded 18 or reference to a constant
    ```

## 5. Performance and Resource Management [Code]

- [Important] Use generators for large datasets or sequences [General]
    ```python
    # Good example - memory efficient
    def process_large_file(filename: str):
        try:
            with open(filename, 'r') as f:
                for line_num, line in enumerate(f): # Process one line at a time
                    yield f"Processed line {line_num}: {line.strip()}"
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")

    # Usage example
    for processed_line in process_large_file("large_log.txt"):
        print(processed_line)

    # Example to avoid - loads everything into memory
    def process_large_file_bad(filename: str) -> list[str]:
        try:
            with open(filename, 'r') as f:
                lines = f.readlines() # Loads all lines
            return [f"Processed line {i}: {line.strip()}" for i, line in enumerate(lines)]
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
            return []
    ```

- [Important] Choose appropriate data structures [General]
    - **List (`list`)**: Ordered, mutable sequence. For frequent additions/removals. O(1) indexing but O(n) for element search (`in`).
    - **Tuple (`tuple`)**: Ordered, immutable sequence. For immutability, use as dictionary keys.
    - **Dictionary (`dict`)**: Key-value pairs. For fast lookups by key (average O(1)). Preserves insertion order in Python 3.7+.
    - **Set (`set`)**: Collection of unique elements. For fast existence checks (`in`, average O(1)), removing duplicates, or set operations (union, intersection, difference).

- [Important] Use appropriate concurrency/parallelism for CPU-bound and I/O-bound operations [App]
    ```python
    import time
    import asyncio
    import aiohttp
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

    # CPU-bound tasks - use multiprocessing (ProcessPoolExecutor)
    def cpu_intensive_task(n):
        print(f"Calculating sum up to {n}...")
        total = sum(i*i for i in range(n))
        print(f"Sum for {n} is {total}")
        return total

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cpu_intensive_task, [10**6, 10**6 + 1, 10**6 + 2]))
    print(f"CPU bound results: {results}")


    # I/O-bound tasks (network/disk wait) - use asyncio (or threading/ThreadPoolExecutor)
    async def fetch_url(session, url):
        print(f"Fetching {url}...")
        try:
            async with session.get(url, timeout=10) as response:
                content = await response.text()
                print(f"Fetched {url}, length: {len(content)}")
                return len(content)
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return 0

    async def main_async():
        urls = ["https://www.google.com", "https://www.python.org", "https://httpbin.org/delay/2"]
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
            print(f"Async IO bound results: {results}")

    asyncio.run(main_async())

    # ThreadPoolExecutor for IO bound tasks (alternative to asyncio)
    def fetch_url_sync(url):
        # Synchronous network request (e.g., using requests library)
        # import requests
        # response = requests.get(url)
        # return len(response.text)
        print(f"Simulating fetch {url}...")
        time.sleep(1) # Simulate network latency
        return len(url) * 100 # Dummy result

    with ThreadPoolExecutor(max_workers=5) as executor:
        urls = ["url1", "url2", "url3", "url4", "url5"]
        results = list(executor.map(fetch_url_sync, urls))
    print(f"Thread IO bound results: {results}")

    ```
    *Note: `multiprocessing` has process communication overhead; `asyncio` requires async-compatible libraries; `threading` is ineffective for CPU-bound tasks due to the GIL but works well for I/O-bound tasks.*

- [Recommended] Avoid unnecessary loops and duplicate calculations [General]
    ```python
    # Good example - cache or reuse calculation results
    items = range(5)
    threshold = calculate_expensive_threshold() # Calculate only once
    processed_items = []
    for item in items:
        if item > threshold:
            processed_items.append(process(item))

    # Example to avoid - recalculating in each loop iteration
    processed_items_bad = []
    for item in items:
        # Recalculating threshold each time
        if item > calculate_expensive_threshold():
            processed_items_bad.append(process(item))
    ```

- [Recommended] Always release resources with `try...finally` or context managers (`with`) [General]
    ```python
    import threading
    lock = threading.Lock()

    # Good example (context manager)
    def update_shared_resource(value):
        with lock: # Acquire lock and automatically release when block ends
            shared_resource = value
            # ... other operations ...

    # Good example (try...finally) - for when context managers aren't available
    lock.acquire()
    try:
        shared_resource = value
        # ... other operations ...
    finally:
        lock.release() # Always releases, whether successful or exception occurs
    ```

## 6. Documentation [Code]

- [Required] Write **Google style** docstrings for public APIs (functions, classes, methods) [Library] [App]
    ```python
    from typing import Tuple # For Python <3.9
    # from typing import tuple # Python 3.9+

    def calculate_distance(point1: tuple[float, float], point2: tuple[float, float]) -> float:
        """Calculate the Euclidean distance between two points.

        Args:
            point1: Coordinates of the first point in (x, y) format.
            point2: Coordinates of the second point in (x, y) format.

        Returns:
            The distance between the points.

        Raises:
            TypeError: If the arguments are not of the correct type.
            ValueError: If the coordinate values are invalid.
        """
        if not (isinstance(point1, tuple) and len(point1) == 2 and
                isinstance(point2, tuple) and len(point2) == 2):
            raise TypeError("Points must be tuples of (x, y)")
        try:
            x1, y1 = float(point1[0]), float(point1[1])
            x2, y2 = float(point2[0]), float(point2[1])
        except (ValueError, TypeError) as e:
            raise ValueError("Coordinates must be numeric") from e

        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        return distance
    ```

- [Important] Add inline comments for complex logic [General]
    ```python
    # Good example
    def calculate_tax(amount, rate, deductions):
        # Calculate base tax amount before applying deductions
        base_tax = amount * rate

        # Apply deductions (maximum allowed deduction is 10% of base tax)
        max_allowed_deduction = base_tax * 0.1
        actual_deduction = min(sum(deductions), max_allowed_deduction) # Apply limit

        # Calculate final tax (minimum tax is 0)
        return max(0, base_tax - actual_deduction)
    ```

- [Recommended] Avoid obvious comments when function and parameter names are clear [General]
    ```python
    # Example to avoid - redundant comment
    # Get the user's age
    def get_user_age(user: User) -> int:
        return user.age
    ```

- [Recommended] Include explanatory comments for important design decisions or complex algorithms [General]
    ```python
    # This regex is a simplified version of RFC 5322 that covers common email formats.
    # It doesn't handle edge cases like commented addresses or IP-literal format.
    # Chosen for balance of performance and practical coverage.
    EMAIL_REGEX = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    ```

## 7. Security Implementation [Code]

- [Required] Always validate and sanitize/escape external input data [General]
    ```python
    from markupsafe import escape # Escaping library used in Flask/Jinja2 etc.

    def process_user_comment(comment: str) -> str:
        # Validation: reject comments that are too long
        if len(comment) > 1024:
            raise ValueError("Comment is too long")

        # Sanitization: escape HTML tags to prevent XSS
        safe_comment = escape(comment)

        # ... processing for storing or displaying the safe comment ...
        return safe_comment
    ```

- [Important] Prevent SQL injection [General]
    ```python
    import sqlite3

    db = sqlite3.connect(":memory:")
    cursor = db.cursor()
    # cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    # cursor.execute("INSERT INTO users (name) VALUES ('Alice')")

    user_id = 1 # Assume this comes from external input

    # Good example - using placeholders
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()

    # Good example (named placeholders)
    # cursor.execute("SELECT * FROM users WHERE id = :id", {"id": user_id})

    # Examples to avoid - string formatting or concatenation (dangerous!)
    # cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
    # user_input = "1; DROP TABLE users" # Malicious input example
    # cursor.execute("SELECT * FROM users WHERE id = " + user_input)
    ```
    *Note: ORMs (SQLAlchemy, Django ORM, etc.) typically handle injection protection, but always be careful when writing raw SQL queries.*

- [Important] Don't hardcode secrets (API keys, passwords, etc.) [General]
    ```python
    import os
    from dotenv import load_dotenv # python-dotenv library

    # Load .env file (e.g., .env file with API_KEY=your_secret_key)
    load_dotenv()

    # Good example - load from environment variables or config files
    api_key = os.environ.get("API_KEY")
    db_password = os.environ.get("DB_PASSWORD")

    if not api_key:
        logger.warning("API_KEY environment variable not set.")
        # Error handling if needed

    # Examples to avoid
    # api_key = "supersecretapikey12345" # Don't write directly in code
    # db_password = "password123"
    ```
    *Note: Don't include `.env` files in version control (add to `.gitignore`). In production, use environment variables or secret management services (AWS Secrets Manager, HashiCorp Vault, etc.).*

- [Important] Use proper escaping to prevent Cross-Site Scripting (XSS) in web applications [App]
    ```python
    # Flask/Jinja2 example (auto-escaping is enabled by default)
    # template.html
    # <p>User comment: {{ user_comment }}</p>  # Django Templates example (auto-escaping is also default)
    # <p>User comment: {{ user_comment }}</p>

    # When explicitly escaping (e.g., using markupsafe library)
    from markupsafe import escape
    untrusted_input = "<script>alert('XSS');</script>"
    safe_output = escape(untrusted_input)
    # safe_output becomes "&lt;script&gt;alert(&#39;XSS&#39;);&lt;/script&gt;"

    # Disabling escaping (only when necessary, use with extreme caution)
    # Jinja2: {{ user_html_content|safe }}
    # Django: {{ user_html_content|safe }}
    ```

- [Recommended] Use standard libraries or trusted third-party libraries (e.g., `cryptography`, `passlib`) for encryption and hashing; avoid custom implementations [General]
    ```python
    import hashlib
    import os
    from cryptography.fernet import Fernet # Symmetric encryption example

    # Password hashing (salt + PBKDF2)
    def hash_password(password: str) -> bytes:
        salt = os.urandom(16)
        # Set iterations sufficiently high (e.g., 260000+)
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 260000)
        return salt + key # Store salt concatenated with the hash

    def verify_password(stored_hash: bytes, provided_password: str) -> bool:
        salt = stored_hash[:16]
        stored_key = stored_hash[16:]
        new_key = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 260000)
        return new_key == stored_key # Use hmac.compare_digest for constant-time comparison if needed

    # Recommended: Use a library like passlib
    # from passlib.context import CryptContext
    # pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    # hashed = pwd_context.hash(password)
    # verified = pwd_context.verify(provided_password, hashed)


    # Data encryption (symmetric)
    # key = Fernet.generate_key() # The key must be securely managed
    # cipher_suite = Fernet(key)
    # data = b"Secret message"
    # encrypted_data = cipher_suite.encrypt(data)
    # decrypted_data = cipher_suite.decrypt(encrypted_data)
    ```

## 8. Test Code [Code]

- [Required] Test function/method names should clearly indicate what's being tested and the expected behavior [General]
    ```python
    # Good examples (pytest style)
    def test_login_with_valid_credentials_returns_success_token():
        # ... test implementation ...
        pass

    def test_login_with_invalid_password_raises_authentication_error():
        # ... test implementation ...
        pass

    def test_get_user_profile_returns_correct_user_data():
        # ... test implementation ...
        pass
    ```

- [Important] Tests should cover normal cases, boundary values, and exception cases [General]
    ```python
    import pytest

    def divide(a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def test_divide_normal_case():
        assert divide(10, 2) == 5

    def test_divide_by_zero_raises_value_error():
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10, 0)

    def test_divide_zero_by_number():
        assert divide(0, 5) == 0

    def test_divide_negative_numbers():
        assert divide(-10, 2) == -5
        assert divide(10, -2) == -5
        assert divide(-10, -2) == 5

    # Boundary value examples (e.g., discount rate calculation)
    def calculate_discount(price, percentage):
         if not (0 <= percentage <= 100):
              raise ValueError("Percentage must be between 0 and 100")
         return price * (percentage / 100)

    def test_discount_boundary_values():
        assert calculate_discount(100, 0) == 0    # Lower bound
        assert calculate_discount(100, 100) == 100 # Upper bound
        assert calculate_discount(100, 50) == 50   # Middle value

    def test_discount_invalid_percentage_raises_error():
        with pytest.raises(ValueError):
            calculate_discount(100, -10) # Below lower bound
        with pytest.raises(ValueError):
            calculate_discount(100, 110) # Above upper bound
    ```

- [Important] Test code should avoid dependencies on external environment (network, DB, filesystem) by using mocks, stubs, and test doubles [General]
    ```python
    from unittest.mock import patch, MagicMock

    # Function to test (depends on external API)
    # def get_weather(city: str) -> str:
    #     import requests
    #     response = requests.get(f"https://api.weather.com/?city={city}")
    #     response.raise_for_status()
    #     return response.json()["forecast"]

    # Good example - mock requests.get
    @patch('requests.get') # Replace 'requests.get' with a mock object
    def test_get_weather_returns_forecast_on_success(mock_get):
        # Mock setup: simulate response object
        mock_response = MagicMock()
        mock_response.json.return_value = {"forecast": "Sunny"}
        mock_response.raise_for_status.return_value = None # No exception
        mock_get.return_value = mock_response # Configure requests.get to return mock_response

        # Test execution
        weather = get_weather("Tokyo")

        # Verification
        assert weather == "Sunny"
        mock_get.assert_called_once_with("https://api.weather.com/?city=Tokyo") # Verify called with correct URL

    @patch('requests.get')
    def test_get_weather_handles_api_error(mock_get):
        # Mock setup: simulate API error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("API Error")
        mock_get.return_value = mock_response

        # Verify exception is raised
        with pytest.raises(requests.exceptions.RequestException, match="API Error"):
            get_weather("London")
    ```

- [Recommended] Structure tests in a readable format (e.g., Arrange-Act-Assert pattern) [Library] [App]
    ```python
    class ShoppingCart:
        def __init__(self): self.items = {}
        def add_item(self, item_name, price, quantity=1): self.items[item_name] = {"price": price, "quantity": quantity}
        def get_total(self): return sum(item["price"] * item["quantity"] for item in self.items.values())

    def test_shopping_cart_get_total_calculates_correctly():
        # Arrange - Setup the test (prepare preconditions)
        cart = ShoppingCart()
        cart.add_item("apple", price=1.0, quantity=5)
        cart.add_item("banana", price=0.5, quantity=2)
        expected_total = (1.0 * 5) + (0.5 * 2) # 5.0 + 1.0 = 6.0

        # Act - Execute the operation being tested
        actual_total = cart.get_total()

        # Assert - Verify the results
        assert actual_total == expected_total
    ```

## 9. Static Analysis and Formatting [Tool]

- [Required] Use static analysis tools and code formatters [General]
    - **`flake8`**: Use to check PEP 8 violations, potential logical errors, and code complexity. Manage project-specific rules in configuration files (`.flake8` or `pyproject.toml`).
    - **`black`**: Use to automatically standardize code formatting. Keep configuration minimal and follow `black`'s default style as a principle.
    - **`isort`**: Use to automatically organize and format import statements. Recommend settings compatible with `black` (`profile = "black"`).
    - **`mypy`**: Use for static type checking based on type hints to detect type-related errors early. Gradual adoption is possible, but aim to eventually cover the main codebase.
    - It's strongly recommended to incorporate these tools into pre-commit hooks in the development environment and into CI/CD pipelines to continuously maintain code quality.
    - *Additional note*: Specific configurations for these tools (rules to enable/disable, line length, `mypy` strictness, etc.) are typically managed in configuration files for each project. This allows for consistent rule application across the project and customization to specific needs. Main configuration files include `pyproject.toml` ([tool.black], [tool.isort], [tool.mypy] sections, etc.), `.flake8`, `.isort.cfg`, and `mypy.ini`.

## Special Markers

Use the following special comment markers to indicate code status:

- Mark legacy code with `# LEGACY` or `#legacy` to indicate areas needing refactoring.
- Mark experimental code with `# EXPERIMENTAL` or `#beta` to indicate unstable or potentially changing code.
- Place third-party code or copied code in dedicated directories and include license information and source attribution.

