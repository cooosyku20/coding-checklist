# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure
- Documentation-based repository containing coding checklists
- Language-specific checklists are placed in appropriate directories (Python, Go)
- Protocol-specific checklists are placed in their own directories (MCP)
- README.md provides a comprehensive overview and navigation for the repository
- Each checklist is linked directly from the README.md for easy access

## Build/Lint/Test Commands
- No specific build commands for this repository as it's documentation-based
- For Python: Follow PEP 8 standards with `flake8` or `black` as linters
- For Go: Use `gofmt` or `go fmt` for formatting, `go vet` for static analysis

## Code Style Guidelines
- Python: 
  - PEP 8 standards with 4-space indentation
  - snake_case for functions/variables, CamelCase for classes
  - Type hints encouraged (Python 3.5+)
  - Prefer f-strings (Python 3.6+)

- Go:
  - Package names: lowercase, no underscores
  - Public identifiers: CamelCase with capital first letter
  - Private identifiers: camelCase with lowercase first letter
  - Error handling with context and wrapping

## Documentation Standards
- Markdown files for all documentation
- Each checklist should have clear sections and subsections
- Keep explanations concise but informative
- Use consistent formatting across all checklist files
- Preserve Japanese content for bilingual compatibility

## Contributing Guidelines
- Follow the contribution process outlined in README.md
- Ensure new content matches the existing structure and formatting
- Maintain consistent naming conventions for new files and directories
- Update README.md with links to any newly added checklists

## Repository Navigation
- Start with README.md for an overview of all available resources
- Each checklist follows a consistent format with similar section organization
- MCP documentation explains the Model Context Protocol for AI model interactions