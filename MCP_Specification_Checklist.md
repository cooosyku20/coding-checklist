# MCP (Model Context Protocol) Server Checklist

The following checklist helps verify whether a server or service is functioning as an MCP server. Items marked as "[Required]" are essential for fulfilling the basic role of an MCP server.

---

## Category 1: Protocol Version and Profile

- Protocol Version Declaration [Required]
  * Explicitly declare the specific version of the MCP protocol supported (e.g., MCP 2025-03-26).
  * If supporting multiple versions, clearly specify compatibility for each version.

- Minimum Implementation Profile Compliance [Required]
  * Comply with the "minimum MCP implementation profile" that includes at least:
    - Processing of basic message types (`initialize`, `initialized`, `shutdown`, `exit`)
    - Basic resource provision functionality (`mcp/getResource`)
    - Basic error handling (implementation of required error codes)
    - Support for at least one transport layer

- Protocol Extension Support [Recommended]
  * Proper handling of standard extensions (including how to respond to unsupported extensions).
  * When implementing custom extensions, adhere to naming conventions to avoid conflicts with standard specifications.

## Category 2: Basic MCP Protocol Compliance

- Protocol Specification Adherence [Required]
  * Strict adherence to MCP core specifications including JSON-RPC based communication and error handling.
  * Compliance with JSON-RPC 2.0 specifications (object structure, field names, data types, etc.).

- Required Message Type Processing [Required]
  * Appropriate processing and response to the following basic message types:
    - `initialize`: Receiving client information and responding with server capabilities
    - `initialized`: Processing initialization completion notification
    - `shutdown`: Session termination processing for normal exit
    - `exit`: Processing for immediate termination
    - `$/cancelRequest`: Processing request cancellations

- Notification Processing [Required]
  * Support for one-way notifications without `id` (e.g., `window/logMessage`, `window/showMessage`).
  * No responses should be returned for notifications.

- JSON-RPC Batch Request Support [Required]
  * Ability to process batch requests that send multiple JSON-RPC requests as a single array.
  * Generation of individual responses for each request in the batch.

- Response Structure Compliance [Required]
  * `"result"` object: Include appropriate response data in the correct format for successful operations.
  * `"error"` object: Include accurate `code`, `message`, and optional `data` fields for failures.
  * Compliance with JSON schema for responses.

## Category 3: Core Functionality (Context Provision)

- Resource Provision Capability [Required]
  * Ability to provide external resources (documents, data, code, etc.) in response to AI client requests.
  * Provide resources in the following formats:
    - Plain text (UTF-8 encoding)
    - Structured data (JSON, YAML, XML, etc.)
    - Binary data (Base64 encoding)

- Tool Execution Capability [Required]
  * Define and expose external functions or operations (API calls, database queries, file writing, etc.) as tools that AI clients can use.
  * Compliance with tool execution protocol:
    - Accurate parameter passing (including type conversion and validation)
    - Return of execution results in standardized format
    - Timeout handling for tool execution
    - Appropriate handling of concurrent execution
  * Implementation of tool annotations (per 2025-03-26 specification update):
    - Providing metadata about tool behavior (e.g., read-only vs. destructive operations)
    - Providing information about intended use cases and constraints

- Prompt Provision Capability [Recommended]
  * Ability to provide prompt templates or standard text to AI clients to assist with task execution or context understanding.
  * Support for version management and parameterization of prompts.

- Method Naming and Schema [Required]
  * Defined method names and JSON schemas for requests/responses, such as:
    - `mcp/getResource`: Retrieving external resources
    - `mcp/executeTool`: Executing tools
    - `mcp/getPrompt`: Retrieving prompt templates
  * Request/response structures that conform to the exact schema for each method.

- Parameter Validation [Required]
  * Verification and enforcement of required fields.
  * Type checking and value range validation for parameters.
  * Rejection of requests containing undefined properties.
  * Appropriate error responses for validation failures (error code -32602).

- Cancellation Support [Required]
  * Processing of cancellation requests (`$/cancelRequest`) for long-running operations.
  * Detailed implementation of cancellation handling:
    - How to respond to cancelled requests (using appropriate error codes)
    - State management for partially completed operations
    - Resource release after cancellation

## Category 4: Security and Access Control

- Authentication, Authorization, and Access Control [Required]
  * Authentication mechanisms to verify legitimate clients for connections, resource access, and tool execution requests.
  * Authorization and access control functionality to limit the scope of resources accessible and tools permitted for each client.
  * Prevention of AI accessing excessive information or performing unauthorized operations.

- OAuth 2.1 Compliance [Required]
  * Implementation of OAuth 2.1 framework for authenticating remote HTTP servers (per 2025-03-26 specification update).
  * Support for the following OAuth flows:
    - Authorization code flow: When the client is acting on behalf of an end user
    - Client credentials flow: When the client is another application
  * Appropriate security measures for both public and confidential clients (including PKCE).
  * Implementation of OAuth 2.0 Authorization Server Metadata (RFC8414).
  * Support for OAuth 2.0 Dynamic Client Registration Protocol (RFC7591) (recommended).

- Secure Data Handling [Required]
  * Appropriate measures against security threats in data processing (especially sensitive information) and external system integration (e.g., unintended data disclosure, unauthorized writing, command injection).
  * Sanitization and validation of input data.
  * Proper implementation of token management (secure storage, validation, renewal).

- Error Code Standards [Required]
  * Use of JSON-RPC error codes for authentication/authorization failures (e.g., -32001 to -32099).
  * Implementation of the following standard error codes:
    - -32700: Parse error
    - -32600: Invalid request
    - -32601: Method not found
    - -32602: Invalid parameters
    - -32603: Internal error
    - -32001: Authentication error
    - -32002: Authorization error
    - -32003: Resource not found
    - -32004: Tool execution error

## Category 5: Technical Implementation Elements

- Supported Transport Layers [Required]
  * Support and normal communication for at least one transport layer defined in the MCP specification.
  * Detailed requirements for each transport layer:
    - stdio: Communication via standard input/output, framing with byte length prefix (for local integration)
    - HTTP/SSE (before 2025-03-26 specification update): HTTP request/response, asynchronous communication via Server-Sent Events
    - Streamable HTTP (per 2025-03-26 specification update): More flexible HTTP-based transport, supporting JSON-RPC batch processing
    - WebSocket: Bidirectional message streams, appropriate handshaking and connection management

- Bidirectional Communication Support [Design Consideration]
  * Design or implementation capability to utilize the bidirectional communication features provided by the MCP protocol to send asynchronous information (e.g., processing progress, event notifications) from server to client as needed. (Not required but effective for advanced feature integration)
  * Implementation of asynchronous notification mechanisms.

- Message Framing Compliance [Required]
  * Adherence to framing (e.g., `Content-Length` header for HTTP, byte length prefix for stdio).
  * Accurate framing implementation for each transport layer:
    - HTTP: Message length specification using Content-Length header
    - stdio: Message boundary definition using binary length prefix (8 bytes)
    - WebSocket: Compliance with WebSocket frame protocol

- JSON-RPC 2.0 Message Format Compliance [Required]
  * Appropriate implementation of the following three message types:
    - Request: Messages expecting a response (including method, id, params)
    - Response: Responses to requests (including id, result or error)
    - Notification: One-way messages not expecting a response (including method, params, but no id)
  * Support for JSON-RPC batch processing (per 2025-03-26 specification update)

## Category 6: Protocol Validation and Conformance

- Protocol Conformance Testing [Required]
  * Conduct and pass MCP protocol conformance tests including:
    - Basic message flow validation (initialize -> initialized -> various operations -> shutdown -> exit)
    - Error handling validation (invalid requests, timeouts, resource constraints)
    - Boundary condition testing (large messages, special characters, long-running operations)

- Capability Discovery [Recommended]
  * Mechanisms for AI clients to programmatically discover and understand through the MCP protocol the types of resources provided by the server, available tools, and their usage (required parameters, expected responses, etc.) (e.g., providing descriptions of features/tools).
  * Provision of self-descriptive API specifications.

- Documentation and Specification Publication [Recommended]
  * Provision of technical documentation on MCP server functionality, supported protocol versions, specific specifications of provided resources and tools, authentication/authorization methods, etc.
  * Publication of API specifications in standard formats (OpenAPI, JSON Schema, etc.).

- Return Server Capabilities on Initialization [Recommended]
  * Return a `serverCapabilities` object that enumerates supported features in the `initialize` response.
  * Specific feature descriptions (supported resource types, available tools, authentication methods, etc.).
