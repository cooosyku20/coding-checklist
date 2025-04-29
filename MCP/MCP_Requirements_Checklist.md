# MCP (Model Context Protocol) Requirements Checklist

The following checklist helps verify whether a server or service functions as an MCP server. Items marked as "[Required]" are essential for fulfilling the basic role as an MCP server.

---

## Category 1: Protocol Version and Profiles

- Protocol Version Declaration [Required]
  * Explicitly declare the specific version of the MCP protocol supported (e.g., MCP 2025-03-26).
  * If supporting multiple versions, specify compatibility handling for each version.

- Minimum Implementation Profile Compliance [Required]
  * Comply with the "Minimal MCP Implementation Profile" which includes at least:
    - Processing basic message types (`initialize`, `initialized`, `shutdown`, `exit`)
    - Basic resource provision functionality (`mcp/getResource`)
    - Basic error handling (implementing required error codes)
    - Support for at least one transport layer

- Protocol Extension Support [Recommended]
  * Properly handle standard extensions (including how to respond to unsupported extensions).
  * Follow naming conventions that avoid conflicts with standard specifications when implementing custom extensions.

## Category 2: Basic MCP Protocol Compliance

- Protocol Specification Compliance [Required]
  * Strictly adhere to MCP core specifications including JSON-RPC based communication and error handling.
  * Comply with JSON-RPC 2.0 specification (object structure, field names, data types, etc.).

- Required Message Type Processing [Required]
  * Properly process and respond to these basic message types:
    - `initialize`: Receive client information and respond with server capabilities
    - `initialized`: Process initialization completion notification
    - `shutdown`: Process session termination for normal shutdown
    - `exit`: Process immediate termination
    - `$/cancelRequest`: Process request cancellation

- Notification Processing [Required]
  * Support one-way notifications without `id` (e.g., `window/logMessage`, `window/showMessage`).
  * Do not return responses for notifications.

- JSON-RPC Batch Request Support [Required]
  * Process batch requests that send multiple JSON-RPC requests as a single array.
  * Generate individual responses for each request in the batch.

- Response Structure Compliance [Required]
  * `"result"` object: Include appropriate response data in the correct format for successful operations.
  * `"error"` object: Include accurate `code`, `message`, and optional `data` for failures.
  * Adhere to JSON schema for responses.

- Progress Notification Extensions [Recommended]
  * Add a `message` field to `ProgressNotification` to provide descriptive status updates.
  * Provide messages that clarify specific processing details or stages along with progress.

## Category 3: Core Functionality (Context Provision)

- Resource Provision Capability [Required]
  * Ability to provide external resources (documents, data, code, etc.) in response to AI client requests.
  * Provide resources in these formats:
    - Plain text (UTF-8 encoding)
    - Structured data (JSON, YAML, XML, etc.)
    - Binary data (Base64 encoding)
    - Audio data (with appropriate format and metadata)

- Tool Execution Capability [Required]
  * Define and expose external functions or operations (API calls, database queries, file writes, etc.) as tools that AI clients can use.
  * Comply with tool execution protocol:
    - Accurate parameter passing (including type conversion and validation)
    - Returning execution results in standardized format
    - Handling tool execution timeouts
    - Properly handling concurrent execution
  * Implement tool annotations (per 2025-03-26 specification update):
    - Provide metadata about tool behavior (e.g., read-only vs. destructive operations)
    - Provide information about intended use cases and constraints
    - Consider trustworthiness of tool annotations (annotations should be treated as untrustworthy unless obtained from trusted servers)

- Prompt Provision Capability [Recommended]
  * Ability to provide prompt templates or standard text to AI clients to assist with task execution or context understanding.
  * Support prompt versioning and parameterization.

- Method Naming and Schema [Required]
  * Defined method names and JSON schemas for requests/responses:
    - `mcp/getResource`: Retrieving external resources
    - `mcp/executeTool`: Executing tools
    - `mcp/getPrompt`: Retrieving prompt templates
  * Request/response structures that adhere to precise schemas for each method.

- Parameter Validation [Required]
  * Verification and enforcement of required fields.
  * Type checking and value range validation of parameters.
  * Rejection of requests with undefined properties.
  * Appropriate error responses for validation failures (error code -32602).

- Cancellation Support [Required]
  * Processing cancellation requests (`$/cancelRequest`) for long-running operations.
  * Detailed implementation of cancellation handling:
    - Response method for cancelled requests (using appropriate error codes)
    - State management for partially completed operations
    - Resource release after cancellation

- Argument Auto-completion Support [Recommended]
  * Implementation of `completions` functionality allowing AI clients to suggest auto-completion candidates for arguments.
  * Support for improving efficiency and accuracy of argument input in tools and resource operations.

## Category 4: Security and Access Control

- Authentication, Authorization, Access Control [Required]
  * Authentication mechanism to verify legitimate clients for connections, resource access, and tool execution requests.
  * Authorization and access control features to limit each client's access to resources and permissible tools.
  * Prevention of AI access to excessive information or performing unauthorized operations.

- OAuth 2.1 Compliance [Required]
  * Implementation of OAuth 2.1 framework for authenticating remote HTTP servers (per 2025-03-26 specification update).
  * Support for these OAuth flows:
    - Authorization code flow: When client acts on behalf of end users
    - Client credentials flow: When client is another application
  * Appropriate security measures for both public and confidential clients (including PKCE).
  * Implementation of OAuth 2.0 Authorization Server Metadata (RFC8414).
  * Support for OAuth 2.0 Dynamic Client Registration Protocol (RFC7591) (recommended).

- Secure Data Handling [Required]
  * Appropriate countermeasures against security threats in data processing and external system integration (e.g., unintended data disclosure, unauthorized writes, command injection).
  * Sanitization and validation of input data.
  * Proper token management implementation (secure storage, validation, renewal).

- Error Code Standards [Required]
  * Use of JSON-RPC error codes for authentication/authorization failures (e.g., -32001 to -32099).
  * Implementation of these standard error codes:
    - -32700: Parse error
    - -32600: Invalid request
    - -32601: Method not found
    - -32602: Invalid params
    - -32603: Internal error
    - -32001: Authentication error
    - -32002: Authorization error
    - -32003: Resource not found
    - -32004: Tool execution error

- User Consent and Tool Execution Transparency [Required]
  * Mechanism to obtain explicit consent from users before tool execution.
  * Clear information about tool behavior and potential impacts.
  * Implementation of confirmation processes before execution, especially for destructive operations.

## Category 5: Technical Implementation Elements

- Supported Transport Layers [Required]
  * Support and normal communication for at least one transport layer defined in the MCP specification.
  * Detailed requirements for each transport layer:
    - stdio: Communication via standard input/output, framing with byte length prefix (for local integration)
    - HTTP/SSE (pre-2025-03-26 specification update): HTTP request/response, asynchronous communication via Server-Sent Events
    - Streamable HTTP (per 2025-03-26 specification update): More flexible HTTP-based transport, support for JSON-RPC batch processing
    - WebSocket: Bidirectional message stream, proper handshake and connection management

- Bidirectional Communication Support [Design Consideration]
  * Design or implementation capability to leverage bidirectional communication features provided by the MCP protocol to send asynchronous information (e.g., processing progress, event notifications) from server to client as needed. (Not required, but effective for advanced feature integration)
  * Implementation of asynchronous notification mechanisms.

- Message Framing Compliance [Required]
  * Adherence to framing (e.g., `Content-Length` header for HTTP, byte length prefix for stdio).
  * Accurate framing implementation for each transport layer:
    - HTTP: Message length specification via Content-Length header
    - stdio: Message boundary definition via binary length prefix (8 bytes)
    - WebSocket: Compliance with WebSocket frame protocol

- JSON-RPC 2.0 Message Format Compliance [Required]
  * Proper implementation of these three message types:
    - Requests: Messages expecting responses (containing method, id, params)
    - Responses: Responses to requests (containing id, result or error)
    - Notifications: One-way messages not expecting responses (containing method, params, but no id)
  * Support for JSON-RPC batch processing (per 2025-03-26 specification update)

## Category 6: Client Capabilities and Extensions

- Extended Client Capability Support [Recommended]
  * Implementation of additional client capabilities to enhance MCP connections:
    - Roots: Hierarchical management of multiple resources or contexts
    - Sampling: Support for efficient sampling by language models
  * Server-side extension implementations leveraging these capabilities.

- Client-Server Capability Negotiation [Required]
  * Client and server capability declaration and mutual understanding during initialization phase.
  * Explicit declaration of available protocol features and primitives during the session.
  * Standard protocol extension mechanism for capability expansion.

## Category 7: Protocol Validation and Conformance

- Protocol Conformance Testing [Required]
  * Conduct and pass MCP protocol conformance tests including:
    - Basic message flow validation (initialize -> initialized -> various operations -> shutdown -> exit)
    - Error handling validation (invalid requests, timeouts, resource constraints)
    - Boundary condition testing (large messages, special characters, long-running operations)

- Functionality Discovery [Recommended]
  * Mechanism for AI clients to programmatically discover and understand the types of resources the server provides, available tools, and how to use them (required parameters, expected responses, etc.) through the MCP protocol (e.g., providing descriptions of features/tools).
  * Provision of self-descriptive API specifications.

- Documentation and Specification Publication [Recommended]
  * Provision of technical documentation regarding MCP server functionality, supported protocol versions, specific resource or tool specifications, authentication/authorization methods, etc.
  * Publication of API specifications in standard formats (OpenAPI, JSON Schema, etc.).

- Returning Server Capabilities in Initialization [Recommended]
  * Return a `serverCapabilities` object in the `initialize` response enumerating supported capabilities.
  * Specific capability descriptions (supported resource types, available tools, authentication methods, etc.).
