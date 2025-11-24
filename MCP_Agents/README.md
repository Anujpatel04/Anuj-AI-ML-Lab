# MCP Agents

Collection of Model Context Protocol (MCP) based agents for various automation and analysis tasks.

## Projects

### Browser MCP Agent

Browser automation application that allows you to control a web browser using natural language commands.

- Uses Azure OpenAI for command interpretation
- Playwright for browser automation
- Supports navigation, clicking, scrolling, and content extraction

See `Browser_mcp_agent/README.md` for setup and usage instructions.

### GitHub MCP Agent

GitHub repository analysis tool that allows you to explore and analyze GitHub repositories using natural language queries.

- Uses OpenAI API for query processing
- GitHub MCP Server for repository access
- Supports issues, pull requests, and repository analysis

See `github_mcp_agent/README.md` for setup and usage instructions.

## Requirements

- Python 3.11+ (for Browser MCP Agent)
- Python 3.8+ (for GitHub MCP Agent)
- Node.js and npm (for Playwright in Browser MCP Agent)
- Docker (for GitHub MCP Agent)

## Configuration

Each agent has its own configuration files:
- `mcp_agent.config.yaml` - Agent configuration
- `mcp_agent.secrets.yaml` - API keys and secrets (not committed to git)
