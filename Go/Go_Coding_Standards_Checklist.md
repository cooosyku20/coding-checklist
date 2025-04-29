# Go Coding Standards Checklist

## Legend

**Application Context**:
- [General]: All Go code
- [Script]: Single programs or command line tools
- [Library]: Reusable packages/libraries
- [App]: Large applications/services

**Priority**:
- [Required]: Essential for basic code quality and security
- [Important]: Significantly improves code maintainability, readability, and safety
- [Recommended]: Best practices for better code and processes

**Category**:
- [Code]: Items related to writing code
- [Design]: Items related to code design and structure
- [Tools]: Items related to tools and automation
- [Security]: Items related to security
- [Process]: Items related to development process

## 1. Idiomatic Go Code Style [Code]

- [Required] Follow Go standard format [General]
    - Use `gofmt` / `go fmt`. Editor integration and CI checks recommended.
- [Required] Adhere to Go naming conventions [General]
    - Package names: lowercase, no underscores (`httputil`). Name should indicate package contents.
    - Exported identifiers: Start with capitals (`PublicFunction`, `UserService`).
    - Unexported identifiers: Start with lowercase (`privateFunction`, `userService`).
    - Interface names: `Reader`, `Writer`, etc. (often with `er` suffix). For single method interfaces: method name+er (`Reader` for `Read()`).
    - Acronyms: Be consistent (`UserID` or `userId` - pick one, `HTTPClient`, etc.). Common practice is all capitals (`ID`, `URL`, `HTTP`).
- [Required] Handle errors effectively [General]
    ```go
    // Good example: Wrap with %w and add context (Go 1.13+)
    if err != nil {
        return fmt.Errorf("failed to process request: %w", err)
    }
    // Good example: Early return
    if err := validate(input); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }
    ```
- [Important] Use context appropriately [General]
    - Functions performing I/O, external API calls, or request-scoped processing should take `context.Context` as first argument.
    - Use appropriately for timeout and cancellation propagation.
    ```go
    func ProcessRequest(ctx context.Context, req *Request) (*Response, error) {
        // Create sub-context with timeout
        timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
        defer cancel() // Don't forget to call cancel()

        data, err := fetchData(timeoutCtx, req.ID) // Pass context to sub-operations
        // ...
    }
    ```
- [Important] Leverage zero values [General]
    - Design struct fields so zero values (`0`, `false`, `""`, `nil`, etc.) represent meaningful default states.
    ```go
    type Config struct { MaxRetries int } // Zero value is 0
    func getMaxRetries(cfg Config) int {
        if cfg.MaxRetries == 0 { return 3 } // Use default for zero value
        return cfg.MaxRetries
    }
    ```
- [Important] Write efficient code [General]
    - Don't use pointers unnecessarily for method receivers or function arguments with small value types or structs (when copy cost is low).
    - Consider pointers for large structs to avoid copy cost.
    ```go
    func ProcessValue(v Value) Result { /* ... */ } // Pass by value for small types
    func ProcessLargeStruct(s *LargeStruct) Result { /* ... */ } // Pass by pointer for large types
    ```
- [Recommended] Keep code clear and concise [General]
    - Avoid complex conditionals or deep nesting.
    - Use early returns (`guard clause`) to reduce nesting.
    ```go
    // Good example: Early return
    func checkPermissions(user *User, resource *Resource) error {
        if user == nil { return errors.New("user is nil") }
        if !user.IsActive { return errors.New("user is inactive") }
        if !user.CanAccess(resource) { return errors.New("permission denied") }
        // ... processing for permitted case ...
        return nil
    }
    ```

## 2. Code Structure and Design [Design]

- [Required] Apply the Single Responsibility Principle [General]
    - Design functions, methods, and types (structs, interfaces) to have a single, clear responsibility.
    - Consider splitting large functions or types.
    ```go
    // Good example: Separation of responsibilities
    type UserRepo struct { db *sql.DB } // DB operations
    func (r *UserRepo) GetByID(id string) (*User, error) { /* ... */ }
    type AuthService struct { repo UserRepo } // Authentication logic
    func (s *AuthService) Authenticate(u, p string) (bool, error) { /* ... */ }
    ```
- [Required] Accept interfaces, return structs [General]
    - For function/method arguments, use interface types rather than concrete types to increase flexibility and testability.
    - Constructor functions (`NewXxx`) typically return a pointer to a concrete struct.
    ```go
    // Good example
    type Cache interface { Get(key string) (any, bool); Set(key string, val any) }
    func NewHandler(cache Cache) *Handler { /* ... */ } // Accept interface

    type Handler struct{ cache Cache }
    func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) { /* ... */ }
    ```
- [Important] Practice separation of concerns [Library] [App]
    - Clearly separate different aspects of applications (e.g., UI/API, business logic, data access, external service integration).
    - Consider design patterns like layered architecture, hexagonal architecture, etc.
- [Important] Design small, focused packages [Library] [App]
    - Focus packages on the responsibility indicated by their name.
    - Design package structure to avoid circular dependencies.
    - Be cautious about creating generic packages like `util`, `common`, or `helper` and avoid letting them grow too large.
- [Important] Maintain appropriate dependency direction [App]
    - Dependencies should flow in one direction (e.g., from more volatile layers to more stable layers).
    - Typically follows a pattern like `handler` -> `service` -> `repository` -> `domain`.
- [Recommended] Define interfaces at the point of use [General]
    - The code using a functionality should define the minimal interface it needs. This reduces unnecessary dependencies on implementation details.
    ```go
    // Consumer side (e.g., service package)
    package service
    type UserNotifier interface { Notify(userID string, message string) error }
    type UserService struct { notifier UserNotifier }
    // ...

    // Implementation side (e.g., email package)
    package email
    type EmailSender struct { /* ... */ }
    func (s *EmailSender) Notify(userID string, message string) error { /* Send email */ }
    ```
- [Recommended] Prefer composition over inheritance [General]
    - Go doesn't have class inheritance. Reuse and extend functionality through struct embedding or by having fields of other types.
    ```go
    // Good example: Struct embedding
    type BaseHandler struct { logger *log.Logger }
    func (h *BaseHandler) Log(msg string) { h.logger.Print(msg) }
    type UserHandler struct { BaseHandler; db *sql.DB } // Embed BaseHandler
    func (h *UserHandler) Handle() { h.Log("Handling user request...") } // Use embedded method

    // Good example: Having fields of other types
    type TaskProcessor struct { queue JobQueue; worker Worker }
    ```

## 3. Error Handling and Logging [Code]

- [Required] Treat errors as values and propagate appropriately [General]
    - Don't ignore errors, check them (`if err != nil`).
    - When returning errors to callers, add context and wrap the original error using `fmt.Errorf` with `%w` (Go 1.13+).
    ```go
    func readFile(path string) ([]byte, error) {
        data, err := os.ReadFile(path)
        if err != nil {
            return nil, fmt.Errorf("readFile %s: %w", path, err) // Wrap with %w
        }
        return data, nil
    }
    ```
- [Important] Use custom error types or error variables to distinguish error types [General]
    - Define package-level error variables (sentinel errors) or custom error types to distinguish specific error conditions.
    - Use `errors.Is` (value comparison) or `errors.As` (type comparison and value retrieval) to check errors.
    ```go
    // Sentinel error
    var ErrResourceNotFound = errors.New("resource not found")
    // Custom error type
    type NetworkError struct { Host string; Err error }
    func (e *NetworkError) Error() string { /* ... */ }
    func (e *NetworkError) Unwrap() error { return e.Err } // For errors.Is/As to traverse wrapped errors

    // Caller
    err := action()
    if errors.Is(err, ErrResourceNotFound) { /* ... */ }
    var netErr *NetworkError
    if errors.As(err, &netErr) { /* ... netErr.Host ... */ }
    ```
- [Important] Keep error handling concise, check errors only once [General]
    - Follow the pattern `if err != nil { return err }` and use early returns.
    - Don't deeply nest the "happy path" code.
- [Important] Use structured logging for consistent log output [App]
    - Record not just log messages but also related information (request ID, user ID, error info, etc.) as key-value pairs (`JSON`, `logfmt`, etc.).
    - Consider libraries like `log/slog` (Go 1.21+), `logrus`, `zap`.
    ```go
    // slog example (Go 1.21+)
    slog.Error("Failed to process order", "order_id", orderID, "error", err)
    ```
- [Important] Use panics sparingly and always recover (within recoverable scopes) [General]
    - Use panics only for unrecoverable states (e.g., fatal errors during initialization) or programmer bugs (e.g., nil pointer dereference, out-of-bounds index access).
    - Use `defer` and `recover` to catch panics at top levels like public APIs, request handlers, or Goroutine entry points, and perform appropriate logging and error handling.
    ```go
    // defer + recover example
    defer func() {
        if r := recover(); r != nil {
            log.Printf("Recovered from panic: %v\n%s", r, debug.Stack())
            // Error response or resource cleanup
        }
    }()
    ```
- [Recommended] Use sentinel errors (`io.EOF`, etc.) effectively [General]
    - Understand the meaning of sentinel errors defined in standard libraries (`io.EOF`, `sql.ErrNoRows`, etc.) and handle them appropriately.
    - When defining your own sentinel errors, make their intent clear.

## 4. Concurrency [Code]

- [Required] Manage Goroutines properly and prevent leaks [General]
    - Use `sync.WaitGroup` to wait for launched Goroutines to complete.
    - When launching Goroutines in loops, pass loop variables correctly (as arguments or by rebinding inside).
    ```go
    var wg sync.WaitGroup
    for i := 0; i < 5; i++ {
        i := i // Capture loop variable
        wg.Add(1)
        go func() {
            defer wg.Done()
            fmt.Println("Worker", i)
        }()
    }
    wg.Wait()
    ```
- [Required] Use context to control Goroutine lifecycle [General]
    - Use `context.Context` to propagate timeouts, deadlines, and cancellation signals between Goroutines.
    - Pass `ctx` to functions doing I/O or blocking operations, and check `ctx.Done()` for early termination.
    - Call the `cancel` function from `context.WithCancel`, `context.WithTimeout`, or `context.WithDeadline` using `defer`.
    ```go
    func longOperation(ctx context.Context) error {
        select {
        case <-time.After(10 * time.Second): // Time-consuming operation
            fmt.Println("Operation finished")
            return nil
        case <-ctx.Done(): // Detect cancellation/timeout
            fmt.Println("Operation cancelled:", ctx.Err())
            return ctx.Err()
        }
    }
    ```
- [Important] Choose appropriate synchronization primitives [General]
    - `sync/atomic`: For low-level atomic operations (counters, flags, etc.).
    - `sync.Mutex`: For exclusive access control to shared resources. Keep critical sections short.
    - `sync.RWMutex`: For shared resource access control when reads vastly outnumber writes.
    - `sync.Once`: For one-time initialization.
    - Channels (`chan`): For communication between Goroutines, synchronization, and passing work.
- [Important] Understand channel usage patterns [General]
    - Unbuffered channels are used for synchronization between send and receive.
    - Buffered channels provide limited asynchrony and flow control.
    - Sending side closes channels, and receiving side detects closure with `for range` or `v, ok := <-ch`.
    - Use `select` statement for non-blocking or timeout-based waiting on multiple channel operations.
- [Important] Implement parallel processing with error handling [General]
    - Establish methods to aggregate errors from multiple Goroutines (error channels, slices protected by `sync.Mutex`, etc.).
    - Consider mechanisms to cancel other related Goroutines when an error occurs in one (`context` or `errgroup` package).
    ```go
    // errgroup example
    g, ctx := errgroup.WithContext(context.Background())
    for _, arg := range args {
        arg := arg
        g.Go(func() error {
            // Use ctx for processing
            return process(ctx, arg) // Returning an error cancels ctx
        })
    }
    if err := g.Wait(); err != nil { /* Handle first error */ }
    ```
- [Recommended] Implement explicit worker pools to control resources [App]
    - Use the worker pool pattern to limit the number of concurrently running Goroutines (e.g., for CPU-bound tasks, external resource access limits).
    - Distribute tasks to workers through channels and fix the number of workers.

## 5. Data and State Management [Code]

- [Required] Prefer immutable data patterns [General]
    - When possible, design methods that return copies with new values rather than methods that modify state. This is particularly advantageous for concurrent processing.
    ```go
    type Config struct { values map[string]string }
    func (c Config) WithValue(key, value string) Config {
        newMap := make(map[string]string, len(c.values)+1)
        for k, v := range c.values { newMap[k] = v }
        newMap[key] = value
        return Config{values: newMap} // Return a new instance
    }
    ```
- [Important] Choose appropriately between value receivers and pointer receivers [General]
    - Pointer receiver (`*T`): When the method needs to modify the receiver's state.
    - Value receiver (`T`): When not modifying state or when the copy cost is negligibly small.
    - Consider pointer receivers (`*T`) for large structs to avoid copy costs.
    - For methods that need to work safely with nil receivers, use pointer receivers (`*T`) with nil checks.
- [Important] Manage maps and slices appropriately [General]
    - Understand that passing `map` or `slice` to a function allows the function to modify the elements (behaves similar to reference types).
    - Always protect concurrent access to `map` with `sync.Mutex` or `sync.RWMutex` as concurrent access can cause fatal data races. Consider using `sync.Map`.
    - When a `slice`'s capacity (`cap`) is exceeded on `append`, a new underlying array is allocated in a different memory area from the original array.
    - Consider specifying initial capacity with `make([]T, length, capacity)` to avoid frequent reallocations in some cases.
- [Important] Use constructor patterns for consistent initialization [General]
    - Provide constructor functions in the form `NewXxx(...)` for struct initialization. This is useful when the zero value is not valid or when initialization logic is required.
    - Consider the functional options pattern for structs with complex configurations or options.
    ```go
    // Functional options pattern example
    type ClientOptions struct { Timeout time.Duration; Retries int }
    type Option func(*ClientOptions)
    func WithTimeout(d time.Duration) Option { /* ... */ }
    func NewClient(opts ...Option) *Client {
        options := ClientOptions{ Timeout: 5*time.Second, Retries: 3 } // Default values
        for _, opt := range opts { opt(&options) } // Apply options
        // ... initialize client ...
    }
    ```
- [Recommended] Use dependency injection to reduce coupling [Library] [App]
    - Have structs hold their dependencies (database connections, loggers, external service clients, etc.) as interface types rather than concrete types.
    - Inject these dependencies through constructors or dedicated methods.
    ```go
    type OrderService struct { db OrderRepository; notifier Notifier }
    func NewOrderService(db OrderRepository, notifier Notifier) *OrderService {
        return &OrderService{db: db, notifier: notifier}
    }
    ```

## 6. Performance and Efficiency [Code]

- [Important] Minimize memory allocations [General]
    - Avoid recreating slices, maps, or concatenating large strings inside loops.
    - Pre-specify size/capacity with `make` for `slice` and `map` when possible.
    - Consider reusing large temporary objects (like buffers) with `sync.Pool`.
    - Use the profiler (`pprof`) to identify unexpected memory allocations.
- [Important] Choose appropriate data structures [General]
    - Select data structures based on access patterns (e.g., `map` for key lookups, `slice` for ordered access).
- [Important] Use profiling for performance optimization [App]
    - Base optimizations on measurements, not assumptions.
    - Create micro-benchmarks with `go test -bench`.
    - Get and analyze CPU, memory, block, goroutine profiles of running applications with `net/http/pprof`.
    - Get detailed execution traces with `runtime/trace`.
- [Important] Optimize I/O efficiency [General]
    - Use the `bufio` package for buffering file and network I/O.
    - Use `io.Copy` for large data transfers.
    - Be mindful of the number of system calls (file operations, network operations, etc.) and consider batch processing when possible.
- [Recommended] Perform efficient string concatenation [General]
    - Use `strings.Builder` for concatenating many strings (especially in loops), as it's most efficient.
    - For a few strings with known count in advance, `+` operator or `fmt.Sprintf` is often fine.
- [Recommended] Design scalable applications [App]
    - Aim for stateless services (keep state in external stores).
    - Appropriately introduce asynchronous processing, queuing, caching.
    - Implement resilience patterns like timeouts, retries, circuit breakers.
    - Introduce rate limiting to prevent overload.

## 7. Documentation [Code]

- [Required] Write GoDoc comments for packages and exported identifiers [General]
    - Write package comments just before the `package` declaration.
    - Write comments (starting with `//`) just before exported types, functions, variables, and constants, explaining their purpose, usage, and any caveats.
    ```go
    // Package cache provides an in-memory caching mechanism.
    package cache

    // Cache stores key-value pairs temporarily. It is safe for concurrent use.
    type Cache struct { /* ... */ }

    // New creates a new Cache with default expiration and cleanup interval.
    func New() *Cache { /* ... */ }

    // Get retrieves a value from the cache. Returns value, true if found, otherwise nil, false.
    func (c *Cache) Get(key string) (interface{}, bool) { /* ... */ }
    ```
- [Important] Use Examples to demonstrate usage [Library]
    - Create functions in `*_test.go` files in the form `ExampleXxx` to show concrete usage examples.
    - Include `// Output:` comments to specify expected output, making the example runnable as a test (`go test`).
    ```go
    func ExampleCache_Get() {
        c := cache.New()
        c.Set("message", "hello")
        val, ok := c.Get("message")
        if ok {
            fmt.Println(val)
        }
        // Output: hello
    }
    ```
- [Recommended] Add explanations for complex logic or design decisions [General]
    - Comment on why a particular algorithm was chosen, why a specific implementation method was used, and provide background or intent that can't be inferred from the code alone.
    - Use `// TODO:` or `// FIXME:` to indicate points for future improvement or fixing.
- [Recommended] Organize packages logically and provide README.md [Library]
    - Include package overview, installation instructions, simple usage examples, and links to documentation (pkg.go.dev) in README.md.

## 8. Testing [Code]

- [Required] Use table-driven tests to test multiple cases [General]
    - Define test cases as a slice of structs and run each as a subtest with `t.Run`.
    ```go
    func TestMyFunc(t *testing.T) {
        testCases := []struct{ name string; input string; want string }{ /* ... cases ... */ }
        for _, tc := range testCases {
            t.Run(tc.name, func(t *testing.T) {
                got := MyFunc(tc.input)
                if got != tc.want { t.Errorf("got %q, want %q", got, tc.want) }
            })
        }
    }
    ```
- [Important] Use mocks or stubs to isolate external dependencies from tests [General]
    - Depend on interfaces for external systems (DB, API, filesystem, etc.), and replace with mock implementations (test doubles) during testing.
    - This makes tests run faster and more reliably, unaffected by external factors.
- [Important] Testing HTTP handlers [App]
    - Create test requests with `NewRequest` from the `net/http/httptest` package, and record responses with `NewRecorder`.
    - Call handler functions or `http.Handler` directly and verify status codes or response bodies recorded in the recorder.
- [Recommended] Create test helper functions to reduce duplicate code [General]
    - Extract common setup or assertion (comparison verification) logic in tests to helper functions.
    - Call `t.Helper()` in helper functions so that test failure reports point to the correct line number in the caller.
- [Recommended] Create benchmark tests to measure performance [Library] [App]
    - For performance-critical areas, create `BenchmarkXxx` functions with the `testing` package (`go test -bench=BenchmarkXxx`).
    - Use `b.ReportAllocs()` to measure memory allocation counts.
- [Recommended] Enable race condition detection (`go test -race`) [General]
    - Always run tests for code with concurrency with the `-race` flag to check for data races. Include this in CI pipelines.

## 9. Code Extensibility and Maintainability [Design]

- [Required] Properly manage module versioning and dependencies [General] [Process]
    - Use Go Modules (`go.mod`, `go.sum`) to clearly manage project dependencies and their versions.
    - Follow semantic versioning (`vMAJOR.MINOR.PATCH`) when publishing libraries.
    - Run `go mod tidy` regularly to keep `go.mod` and `go.sum` up to date.
- [Important] Consider API versioning and backward compatibility [Library] [App]
    - Be mindful of backward compatibility when changing public APIs (library functions/types, HTTP endpoints, etc.).
    - For incompatible changes, increment the major version and allow users to update intentionally (module path `/vN` for libraries, URL path `/api/vN` for HTTP APIs, etc.).
- [Important] Use flags and config to make behavior configurable [App]
    - Make application behavior (listen port, database connection info, external API endpoints, feature flags, etc.) configurable from outside without recompiling code, through config files, environment variables, command-line flags, etc.
    - Leverage the `flag`, `os` packages, or libraries like `spf13/viper`.
- [Recommended] Provide smart extension points [Library]
    - Consider offering extension points like interfaces, function types, or plugin mechanisms (e.g., HTTP middleware, custom strategies) to allow library users to add their own functionality or customize behavior without modifying core functionality.

## 10. Tools, Automation, CI/CD [Tools] [Process]

- [Required] Leverage Go standard tools [General]
    - `gofmt` / `go fmt`: Automatic code formatting. Required.
    - `go vet`: Detect typical errors or suspicious structures in code. Regular execution recommended.
    - `go test`: Execute tests. Also utilize coverage measurement (`-cover`), race detection (`-race`).
- [Important] Adopt advanced static analysis tools and linters [General]
    - `staticcheck`: A high-function static analysis tool that does more checks than `go vet`. Strongly recommended.
    - `golangci-lint`: A tool to run and manage multiple linters (`staticcheck`, `errcheck`, `unused`, etc.) together. Effective to set rules at the project level and run in CI.
- [Important] Check dependency vulnerabilities [General] [Security]
    - `govulncheck`: An official Go tool to detect known vulnerabilities in modules your project depends on. Regular execution (especially in CI/CD) is recommended.
- [Required] Implement automatic checks in CI/CD pipelines [App] / [Important] [Library] [Process]
    - Build mechanisms to automatically run quality checks on code changes (GitHub Actions, GitLab CI, Jenkins, etc.).
    - Example steps to include in pipelines:
        1.  `gofmt -l .` (format check)
        2.  `go vet ./...`
        3.  `staticcheck ./...` or `golangci-lint run`
        4.  `go test -race -cover ./...` (tests, race detection, coverage)
        5.  `govulncheck ./...` (vulnerability scan)
        6.  `go build ./...` (build check)

## 11. Security [Security] [Code]

- [Required] Thoroughly validate input values [App] / [Important] [Library]
    - Never trust external inputs (users, other systems, files, etc.), and always strictly validate and sanitize expected types, formats, lengths, ranges, etc.
    - Implement validation processing with awareness of typical vulnerabilities like SQL injection, XSS, path traversal, etc.
- [Required] Prevent SQL injection [App]
    - Always use placeholders (parameterized queries) when dynamically generating SQL queries. `database/sql` and major ORMs support this. String concatenation for query construction is generally prohibited.
- [Required] Handle credentials securely [App] [General]
    - Store passwords hashed with appropriate algorithms like `bcrypt`. Don't store them as plaintext or with reversible encryption.
    - Don't include secrets like API keys, passwords, or certificates directly in code repositories; manage them via environment variables, dedicated config files (with appropriate permissions), or secret management tools (like Vault).
- [Important] Implement proper authentication and authorization controls [App]
    - Authenticate users reliably for each request and verify if the user has permission for the requested resources or operations.
    - Implement secure session management (hard-to-guess IDs, Secure/HttpOnly/SameSite Cookie attributes, appropriate expiration, etc.).
- [Important] Ensure secure communication [App] [General]
    - Encrypt external communications (HTTPS with clients, external API calls, etc.) using TLS. Don't disable certificate verification (`InsecureSkipVerify`).
- [Important] Prevent information leakage in error handling [App]
    - Don't include information that could provide attack hints (stack traces, internal file paths, database error details, etc.) in error messages returned to end users. Log detailed information instead.
- [Recommended] Set HTTP security headers [App]
    - Properly set headers like `Content-Security-Policy`, `Strict-Transport-Security`, `X-Frame-Options`, `X-Content-Type-Options` to enhance security at the browser level.
- [Recommended] Apply the principle of least privilege [App] [General]
    - Give processes or users only the minimum privileges necessary for their tasks (OS, database, filesystem, etc.).

## 12. Special Markers

- `// LEGACY`: Indicates legacy code that needs refactoring.
- `// EXPERIMENTAL`: Indicates experimental features or APIs that may change in the future.
- Third-party code or copied code should be clearly marked with license and source, and isolated in dedicated directories when possible.