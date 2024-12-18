#!/usr/bin/env python3

import os
import sys
import json
import requests
import signal
import tempfile
import subprocess
from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

class TokenUsage:
    def __init__(self):
        self.total_tokens = 0
        self.cost_per_million = 3.0  # $3 per million tokens

    def add_tokens(self, count: int) -> None:
        self.total_tokens += count

    def get_cost(self) -> float:
        return (self.total_tokens / 1_000_000) * self.cost_per_million

    def get_summary(self) -> str:
        return f"Total tokens: {self.total_tokens:,}\nEstimated cost: ${self.get_cost():.4f}"

class Config:
    def __init__(self):
        self.check_dependencies()
        self.check_environment()
        self.history_file = self.init_history()
        self.token_usage = TokenUsage()

    @staticmethod
    def check_dependencies() -> None:
        try:
            import requests
        except ImportError:
            print("Error: requests package is required. Install with: pip install requests")
            sys.exit(1)

    @staticmethod
    def check_environment() -> None:
        if not os.getenv('ANTHROPIC_API_KEY'):
            print("Error: ANTHROPIC_API_KEY environment variable is not set")
            sys.exit(1)

    @staticmethod
    def init_history() -> str:
        history_dir = Path.home() / '.claude-cli'
        history_dir.mkdir(exist_ok=True)
        history_file = history_dir / 'history.json'
        if not history_file.exists():
            history_file.write_text('[]')
        return str(history_file)

    @staticmethod
    def get_system_prompt() -> str:
        return f"""You are a Linux shell assistant alongside an active bash prompt. Help users write and understand shell commands and scripts.

Key points:
- Always use ```bash code blocks for commands
- Each command block should be self-contained and executable
- Explain what commands do before or after the code blocks
- Multiple command blocks are fine - users can select which to run

The user will execute suggested blocks using a special !run <all | select> command.
They can then share the output with you using a !share command.

Current context:
Directory: {os.getcwd()}"""

class History:
    def __init__(self):
        self.session_history = []

    def test_command(self, command: str, language: str = 'bash') -> Tuple[bool, str]:
        """
        Test if a command has valid syntax without executing it.
        Returns a tuple of (is_valid, error_message).
        """
        if language == 'bash':
            # If it's a single-line command without special chars, consider it valid
            if '\n' not in command and not any(c in command for c in '&|;<>()'):
                return True, ""

            try:
                process = subprocess.run(
                    ['bash', '-n'],
                    input=command,
                    text=True,
                    capture_output=True
                )
                is_valid = process.returncode == 0
                error_msg = process.stderr.strip()
                return is_valid, error_msg
            except Exception as e:
                return False, f"Error checking bash syntax: {str(e)}"

        elif language == 'python':
            try:
                compile(command, '<string>', 'exec')
                return True, ""
            except Exception as e:
                return False, str(e)

        return False, f"Unsupported language: {language}"

    def extract_commands(self, text: str, validate: bool = True) -> List[Tuple[str, str, str]]:
        """
        Extract and optionally validate commands from code blocks.
        Returns a list of tuples: (command, language, error_message).
        """
        commands = []
        lines = text.split('\n')
        in_block = False
        current_block = []
        current_language = None
        valid_languages = {'bash', 'python'}

        for line in lines:
            stripped = line.strip()

            if stripped.startswith('```') and not in_block:
                # Extract the language specification
                language = stripped[3:].strip().lower()
                if language in valid_languages:
                    in_block = True
                    current_block = []
                    current_language = language
            elif stripped == '```' and in_block:
                if current_block and current_language:
                    command = '\n'.join(current_block).strip()
                    if command:
                        if validate:
                            is_valid, error_msg = self.test_command(command, current_language)
                            commands.append((command, current_language, error_msg))
                        else:
                            commands.append((command, current_language, ""))
                in_block = False
                current_block = []
                current_language = None
            elif in_block:
                current_block.append(line)

        return commands

    def get_valid_commands(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract only the syntactically valid commands.
        Returns a list of tuples: (command, language)
        """
        all_commands = self.extract_commands(text, validate=True)
        return [(cmd, lang) for cmd, lang, err in all_commands if not err]

    def add_interaction(self, message: str, response: str) -> None:
        """Add a user-assistant interaction to the history."""
        self.session_history.append({
            'user': message,
            'assistant': response,
            'commands': self.extract_commands(response)
        })

    def get_messages_for_api(self) -> List[Dict]:
        """Convert history into format suitable for API messages."""
        messages = []
        for interaction in self.session_history[-10:]:  # Limit to last 10 messages
            messages.extend([
                {"role": "user", "content": interaction['user']},
                {"role": "assistant", "content": interaction['assistant']}
            ])
        return messages

    def get_last_commands(self) -> List[Tuple[str, str]]:
        """
        Get commands from the last interaction.
        Returns a list of tuples: (command, language)
        """
        if not self.session_history:
            return []
        last_response = self.session_history[-1].get('assistant', '')
        return self.get_valid_commands(last_response)

    def clear_history(self) -> None:
        self.session_history = []

    def save_conversation(self, filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump(self.session_history, f)

    def load_conversation(self, filename: str) -> None:
        with open(filename, 'r') as f:
            self.session_history = json.load(f)

class API:
    def __init__(self, api_key: str, token_usage: TokenUsage, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
        self.token_usage = token_usage

    def send_message(self, message: str, system: str, previous_messages: List[Dict] = None) -> Tuple[str, int]:
        if previous_messages is None:
            previous_messages = []

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1024,
                    "system": system,
                    "messages": previous_messages + [{"role": "user", "content": message}]
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            # Extract token usage from response
            usage = data.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            total_tokens = input_tokens + output_tokens

            # Update token usage
            self.token_usage.add_tokens(total_tokens)

            return data['content'][0]['text'], total_tokens

        except requests.Timeout:
            return "Error: Request timed out. Please try again.", 0
        except requests.RequestException as e:
            return f"Error: API request failed - {str(e)}", 0

class Executor:
    def __init__(self):
        self.current_process = None
        self.recording_file = None
        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signum, frame):
        if self.current_process:
            print("\nTerminating command...")
            try:
                os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
                self.current_process.wait()
            except ProcessLookupError:
                pass
            except KeyboardInterrupt:
                pass
            finally:
                self.current_process = None
        else:
            signal.signal(signal.SIGINT, signal.default_int_handler)
            raise KeyboardInterrupt

    def execute_command(self, command: Union[str, Tuple[str, str]], capture_output: bool = True) -> Optional[str]:
        """
        Execute a command and optionally capture its output.

        Args:
            command: Either a string (assumed to be bash) or a tuple of (command_string, language)
            capture_output: If True, returns the command output. If False, returns None
                          but still executes interactively.
        """
        # Convert string command to tuple format
        if isinstance(command, str):
            cmd_tuple = (command, 'bash')
        else:
            cmd_tuple = command

        if not cmd_tuple or len(cmd_tuple) != 2:
            print("Invalid command provided")
            return None

        cmd, language = cmd_tuple
        if not cmd:
            print("No command provided")
            return None

        # Create temporary file with appropriate extension
        file_extension = '.py' if language == 'python' else '.sh'
        with tempfile.NamedTemporaryFile(mode='w', suffix=file_extension, delete=False) as tmp:
            tmp.write(cmd)
            tmp_path = tmp.name

        output = None
        self.recording_file = None

        try:
            os.chmod(tmp_path, 0o755)
            env = os.environ.copy()
            env.pop('ANTHROPIC_API_KEY', None)

            print(f"\nExecuting {language} command:\n{cmd}\n---")

            # Create the execution command based on language
            if language == 'python':
                exec_cmd = f'python {tmp_path}'
            else:  # bash
                exec_cmd = f'bash {tmp_path}'

            # Always use script for recording
            with tempfile.NamedTemporaryFile(mode='w', suffix='.typescript', delete=False) as rec:
                self.recording_file = rec.name

            script_cmd = ['script', '-q', '-c', exec_cmd, self.recording_file]

            self.current_process = subprocess.Popen(
                script_cmd,
                env=env,
                preexec_fn=os.setsid
            )

            try:
                self.current_process.wait()
                if self.current_process.returncode != 0 and self.current_process.returncode != -signal.SIGTERM:
                    print(f"Command failed with exit code {self.current_process.returncode}")

                # Only read and return the recorded output if capture_output is True
                if capture_output and self.recording_file and os.path.exists(self.recording_file):
                    with open(self.recording_file, 'r', errors='replace') as f:
                        output = f.read()

            except KeyboardInterrupt:
                self.handle_interrupt(signal.SIGINT, None)

            return output

        finally:
            try:
                os.unlink(tmp_path)
                if self.recording_file and os.path.exists(self.recording_file):
                    os.unlink(self.recording_file)
            except OSError:
                pass
            print("---")
            self.current_process = None
            self.recording_file = None

class StyledCLI:
    def __init__(self):
        self.console = Console()
        self.style = Style.from_dict({
            'prompt': '#666666',    # Subtle gray for brackets
            'username': '#00A67D',  # Anthropic green for "claude"
            'at': '#666666',        # Gray separator
            'path': '#87CEEB',      # Sky blue for path
            'arrow': '#00A67D',     # Anthropic green for prompt arrow
            'tokens': '#FFA500',    # Orange color for token info
        })

        self.config = Config()
        self.api = API(os.getenv('ANTHROPIC_API_KEY'), self.config.token_usage)
        self.history = History()
        self.executor = Executor()
        self.command_outputs = []

        history_dir = Path.home() / '.claude-cli'
        history_dir.mkdir(exist_ok=True)

        self.session = PromptSession(
            history=FileHistory(str(Path.home() / '.claude-cli' / 'command_history')),
            style=self.style
        )

    def get_styled_prompt(self) -> HTML:
        username = os.getenv('USER', 'user')
        hostname = os.uname().nodename
        cwd = os.getcwd()
        home = str(Path.home())

        # Replace home directory with ~
        if cwd.startswith(home):
            cwd = '~' + cwd[len(home):]

        return HTML(
            '<prompt>'
            '?<username>claude</username>'
            '<at>:</at>'
            '<path>{}</path>?'
            '<arrow>?</arrow> '
            '</prompt>'.format(cwd)
        )

    def print_welcome(self):
        welcome_text = Text()
        welcome_text.append("Claude CLI", style="bold cyan")
        welcome_text.append(" - Interactive Mode\n\n", style="dim")

        commands = [
            ("!exit", "Exit the program"),
            ("!clear", "Clear current session history"),
            ("!run [n]", "Run command block n (default: 1)"),
            ("!run all", "Run all command blocks"),
            ("!run select", "Choose command block interactively"),
            ("!bash / !python", "Start a bash or python interactive session"),
            ("!share", "Share last command output with Claude"),
            ("!save <file>", "Save current session to file"),
            ("!load <file>", "Load conversation from file"),
            ("!tokens", "Show token usage and cost"),
            ("Ctrl+C", "Stop running command"),
            ("Ctrl+D", "Exit the program"),
            ("Up/Down", "Navigate command history")
        ]

        max_cmd_length = max(len(cmd[0]) for cmd in commands)
        help_text = Text()
        help_text.append("Commands:\n", style="bold yellow")

        for cmd, desc in commands:
            help_text.append(f"  {cmd:<{max_cmd_length+2}}", style="cyan")
            help_text.append(f"{desc}\n", style="white")

        panel = Panel(
            help_text,
            title="[bold]Available Commands",
            border_style="blue"
        )

        self.console.print(welcome_text)
        self.console.print(panel)

    def print_command_output(self, command: str, output: str):
        self.console.print(
            Panel(
                Text(output),
                title=f"[bold blue]Command Output: [white]{command}",
                border_style="blue"
            )
        )

    def print_error(self, message: str):
        self.console.print(f"[bold red]Error:[/bold red] {message}")

    def print_success(self, message: str):
        self.console.print(f"[bold green]Success:[/bold green] {message}")

    def handle_command(self, cmd: str) -> bool:
        try:
            if cmd.startswith("!run"):
                parts = cmd.split(maxsplit=1)
                block_spec = parts[1] if len(parts) > 1 else "1"
                self.handle_run_command(block_spec)
                return True

            elif cmd == "!exit":
                print("\nFinal Usage:")
                print(self.config.token_usage.get_summary())
                sys.exit(0)
            elif cmd == "!tokens":
                print("\nCurrent Usage:")
                print(self.config.token_usage.get_summary())
                return True
            elif cmd == "!clear":
                self.history.clear_history()
                self.command_outputs.clear()
                print("History cleared")
                return True
            elif cmd.startswith("!save "):
                filename = cmd.split(maxsplit=1)[1]
                self.history.save_conversation(filename)
                return True
            elif cmd.startswith("!load "):
                filename = cmd.split(maxsplit=1)[1]
                self.history.load_conversation(filename)
                return True
            elif cmd.startswith("!bash "):
                command = cmd[6:].strip()  # Remove !bash prefix
                self.run_bash_command(command)
                return True
            elif cmd == "!bash":
                self.run_bash_command()  # Start interactive bash session
                return True
            elif cmd.startswith("!python "):
                command = cmd[8:].strip()  # Remove !python prefix
                self.run_python_command(command)
                return True
            elif cmd == "!python":
                self.run_python_command()  # Start interactive python session
                return True

            elif cmd.startswith("!share"):
                parts = cmd.split(maxsplit=1)
                additional_context = parts[1] if len(parts) > 1 else ""

                if self.command_outputs:
                    # Format multiple outputs with block numbers
                    formatted_outputs = "\n\n".join([
                        f"Output from block {block_num}:\n```\n{output}\n```"
                        for block_num, output in self.command_outputs
                    ])

                    # Combine the outputs with any additional context
                    message = f"Here is the output from my commands:\n\n{formatted_outputs}"
                    if additional_context:
                        message = f"{additional_context}\n\n{message}"

                    response, tokens = self.api.send_message(
                        message,
                        self.config.get_system_prompt(),
                        self.history.get_messages_for_api()
                    )
                    print(response)
                    print(f"\n[tokens]Tokens used in this interaction: {tokens:,}[/tokens]")
                    self.history.add_interaction(
                        f"Command outputs with context: {additional_context}\n{formatted_outputs}",
                        response
                    )
                else:
                    print("No command outputs to share")
                return True

            return False

        except KeyboardInterrupt:
            print("\nCommand cancelled")
            return True

    def handle_run_command(self, block_spec: str = "1") -> None:
        """
        Handle the !run command with various arguments:
        - number: Run specific block
        - 'all': Run all blocks
        - 'select': Interactively select block
        - raw command: Execute as bash command
        """
        commands = self.history.get_last_commands()
        if not commands:
            print("No commands found in last response")
            return

        self.command_outputs.clear()

        # Handle "all" command blocks
        if block_spec == "all":
            self._run_all_blocks(commands)
            return

        # Handle interactive selection
        if block_spec == "select":
            self._run_interactive_selection()
            return

        # Handle numeric block selection
        try:
            if block_spec.isdigit():
                block_num = int(block_spec)
                if 1 <= block_num <= len(commands):
                    output = self.executor.execute_command(commands[block_num - 1], capture_output=True)
                    if output:
                        self.command_outputs.append((block_num, output))
                else:
                    print(f"Block number {block_spec} out of range")
                return
        except ValueError:
            pass

        # If we get here, treat as direct bash command
        is_valid, error_msg = self.history.test_command(block_spec)
        if is_valid:
            output = self.executor.execute_command((block_spec, 'bash'), capture_output=True)
            if output:
                self.command_outputs.append((-1, output))
        else:
            print(f"Invalid bash command: {error_msg}")

    def _run_all_blocks(self, commands: List[Tuple[str, str]]) -> None:
        """Execute all command blocks in sequence."""
        for i, cmd in enumerate(commands, 1):
            print(f"\nExecuting block {i}:")
            try:
                output = self.executor.execute_command(cmd, capture_output=True)
                if output:
                    self.command_outputs.append((i, output))
            except KeyboardInterrupt:
                print("\nExecution stopped by user")
                break

    def _run_interactive_selection(self) -> None:
        """Handle interactive block selection from history."""
        history_index = len(self.history.session_history) - 1

        while history_index >= 0:
            # Get commands for current history entry
            if history_index == len(self.history.session_history) - 1:
                commands = self.history.get_last_commands()
                context = "Current response"
            else:
                interaction = self.history.session_history[history_index]
                commands = self.history.get_valid_commands(interaction['assistant'])
                user_msg = interaction['user']
                context = (user_msg[:50] + '...') if len(user_msg) > 50 else user_msg

            if commands:
                print(f"\n{context}:")
                for i, cmd in enumerate(commands, 1):
                    cmd_str, lang = cmd
                    print(f"\nBlock {i} ({lang}):\n{cmd_str}")

                try:
                    selection = input("\nSelect block number (or press Enter for older commands, 'q' to cancel): ").strip().lower()

                    if selection == 'q':
                        print("Selection cancelled")
                        break
                    elif selection == '':
                        history_index -= 1
                        continue

                    try:
                        block_num = int(selection)
                        if 1 <= block_num <= len(commands):
                            output = self.executor.execute_command(commands[block_num-1], capture_output=True)
                            if output:
                                self.command_outputs.append((block_num, output))
                            break
                        else:
                            print(f"Block number {selection} out of range")
                    except ValueError:
                        print(f"Invalid input: {selection}")

                except KeyboardInterrupt:
                    print("\nSelection cancelled")
                    break
            else:
                history_index -= 1

        if history_index < 0:
            print("No more commands found in history")

    def run_bash_command(self, command: Optional[str] = None) -> None:
        """Run a bash command or start an interactive bash session."""
        env = os.environ.copy()
        env.pop('ANTHROPIC_API_KEY', None)  # Don't expose API key to shell

        if command and command.trim():
            print(f"Running bash command: {command}")
            try:
                output = self.executor.execute_command(command, capture_output=True)
                if output:
                    self.command_outputs.append((-1, output))
                    self.history.add_interaction(
                        f"!bash {command}",
                        f"Bash command output:\n```\n{output}\n```"
                    )
            except Exception as e:
                error_msg = f"Error executing command: {e}"
                print(error_msg)
                self.history.add_interaction(
                    f"!bash {command}",
                    f"Error:\n```\n{error_msg}\n```"
                )
        else:
            print("\nStarting interactive bash session (type 'exit' to return to Claude CLI)")
            print("---")

            with tempfile.NamedTemporaryFile(mode='w', suffix='.typescript', delete=False) as typescript:
                try:
                    shell = os.environ.get('SHELL', '/bin/bash')
                    subprocess.run(['script', typescript.name, '-q'], env=env)

                    with open(typescript.name, 'r', errors='replace') as f:
                        session_output = f.read()

                    self.history.add_interaction(
                        "!bash (interactive session)",
                        f"Bash session transcript:\n```\n{session_output}\n```"
                    )

                except KeyboardInterrupt:
                    print("\nBash session terminated")
                    with open(typescript.name, 'r', errors='replace') as f:
                        session_output = f.read()
                    self.history.add_interaction(
                        "!bash (interactive session - interrupted)",
                        f"Interrupted bash session transcript:\n```\n{session_output}\n```"
                    )
                finally:
                    os.unlink(typescript.name)
            print("---")

    def interactive_mode(self):
        self.print_welcome()

        while True:
            try:
                user_input = self.session.prompt(
                    lambda: self.get_styled_prompt()
                ).strip()

                if not user_input:
                    continue

                if not self.handle_command(user_input):
                    # Get previous messages for context
                    previous_messages = self.history.get_messages_for_api()

                    # Send message with conversation history
                    response, tokens = self.api.send_message(
                        user_input,
                        self.config.get_system_prompt(),
                        previous_messages
                    )
                    print(response)

                    # Print token usage for this interaction
                    print(f"\n[tokens]Tokens used in this interaction: {tokens:,}[/tokens]")

                    self.history.add_interaction(user_input, response)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use <Ctrl>+D or !exit[/yellow]")
            except EOFError:
                print("\nFinal Usage:")
                print(self.config.token_usage.get_summary())
                self.console.print("\n[green]Goodbye![/green]")
                sys.exit(0)
            except Exception as e:
                self.print_error(str(e))

    def single_message_mode(self, message: str):
        previous_messages = self.history.get_messages_for_api()
        response, tokens = self.api.send_message(
            message,
            self.config.get_system_prompt(),
            previous_messages
        )
        print(response)
        print(f"\nTokens used: {tokens:,}")
        print(self.config.token_usage.get_summary())

def main():
    cli = StyledCLI()

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage: claude-cli [message]")
            print("  No arguments: Enter interactive mode")
            print("  -h, --help:  Show this help message")
            sys.exit(0)
        else:
            cli.single_message_mode(' '.join(sys.argv[1:]))
    else:
        cli.interactive_mode()

if __name__ == "__main__":
    main()
