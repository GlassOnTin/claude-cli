# Claude CLI

A command-line interface for interacting with Anthropic's Claude AI assistant, specifically designed for shell command assistance and execution.

## Features

- Interactive shell environment with command history
- Direct communication with Claude AI
- Execute shell commands from Claude's suggestions
- Share command outputs back with Claude for analysis
- Save and load conversation history
- Process management with graceful interruption

## Prerequisites

- Python 3.x
- Anthropic API key
- `requests` Python package

## Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install requests prompt_toolkit
   ```
3. Set up your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```

## Usage

### Starting the CLI

```bash
python3 claude-cli.py
```

### Available Commands

- `!exit` - Exit the program
- `!clear` - Clear current session history
- `!run [n]` - Run command block n (default: 1)
- `!run all` - Run all command blocks
- `!run select` - Choose command block interactively
- `!share` - Share last command output with Claude
- `!save <file>` - Save current session to file
- `!load <file>` - Load conversation from file

### Keyboard Shortcuts

- `Ctrl+C` - Stop running command
- `Ctrl+D` - Exit the program
- `Up/Down` - Navigate command history

## Example Usage

1. Ask Claude for help with a command:
   ```
   > How do I find large files in the current directory?
   ```

2. Claude will respond with command suggestions in code blocks

3. Execute suggested commands:
   ```
   !run 1
   ```

4. Share the output with Claude:
   ```
   !share
   ```

## Features in Detail

### Command Execution
- Commands are executed in a controlled environment
- Output is captured and can be shared back with Claude
- Process management handles interruptions gracefully

### History Management
- Conversation history is maintained across sessions
- Save/load functionality for conversation preservation
- Command history with up/down arrow navigation

### Security
- API keys are removed from subprocess environment
- Commands are executed in temporary files
- Process isolation for command execution

## File Structure

The script is organized into several main classes:
- `Config`: Configuration and environment management
- `History`: Conversation and command history
- `API`: Claude API communication
- `Executor`: Command execution and process management
- `CLI`: Main interface and command handling

## License

MIT [License](LICENSE)

EOL
