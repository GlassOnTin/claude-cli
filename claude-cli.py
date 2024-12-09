#!/usr/bin/env python3

import os
import sys
import json
import signal
import tempfile
import subprocess
import requests
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

class Config:
    def __init__(self):
        self.check_dependencies()
        self.check_environment()
        self.history_file = self.init_history()

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

    def extract_commands(self, text: str) -> List[str]:
        """
        Extract commands from various code block formats.
        Handles both ```bash and plain ``` blocks, preserving special characters.
        """
        commands = []
        lines = text.split('\n')
        in_block = False
        current_block = []

        for line in lines:
            stripped = line.strip()
            # Match both ```bash and ``` starts
            if stripped.startswith('```') and not in_block:
                in_block = True
                current_block = []
            elif stripped == '```' and in_block:
                if current_block:
                    # Join and strip to handle any leading/trailing whitespace
                    command = '\n'.join(current_block).strip()
                    if command:  # Only add non-empty commands
                        commands.append(command)
                in_block = False
                current_block = []
            elif in_block:
                # If first line was ```bash, skip it
                if current_block or not stripped == 'bash':
                    current_block.append(line)

        return commands

    def add_interaction(self, message: str, response: str) -> None:
        self.session_history.append({
            'user': message,
            'assistant': response,
            'commands': self.extract_commands(response)
        })

    def get_messages_for_api(self) -> List[Dict]:
        """Convert history into format suitable for API messages"""
        messages = []
        for interaction in self.session_history[-10:]:  # Limit to last 10 messages
            messages.extend([
                {"role": "user", "content": interaction['user']},
                {"role": "assistant", "content": interaction['assistant']}
            ])
        return messages

    def get_last_commands(self) -> List[str]:
        if not self.session_history:
            return []
        return self.session_history[-1].get('commands', [])

    def clear_history(self) -> None:
        self.session_history = []


class API:
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout

    def send_message(self, message: str, system: str, previous_messages: List[Dict] = None) -> str:
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
            return response.json()['content'][0]['text']
        except requests.Timeout:
            return "Error: Request timed out. Please try again."
        except requests.RequestException as e:
            return f"Error: API request failed - {str(e)}"

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

    def execute_command(self, cmd: str, capture_output: bool = True) -> Optional[str]:
        """
        Execute a command and optionally capture its output.

        Args:
            cmd: The command to execute
            capture_output: If True, returns the command output. If False, returns None
                          but still executes interactively.
        """
        if not cmd.strip():
            print("No command provided")
            return None

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as tmp:
            tmp.write(cmd)
            tmp_path = tmp.name

        output = None
        self.recording_file = None

        try:
            os.chmod(tmp_path, 0o755)
            env = os.environ.copy()
            env.pop('ANTHROPIC_API_KEY', None)

            print(f"\nExecuting command:\n{cmd}\n---")

            # Always use script for recording, but only save output if capture_output is True
            with tempfile.NamedTemporaryFile(mode='w', suffix='.typescript', delete=False) as rec:
                self.recording_file = rec.name

            script_cmd = ['script', '-q', '-c', f'bash {tmp_path}', self.recording_file]

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


class CLI:
    def __init__(self):
        self.config = Config()
        self.api = API(os.getenv('ANTHROPIC_API_KEY'))
        self.history = History()  # No longer needs history_file
        self.executor = Executor()
        self.command_outputs = []

        # Keep command history in a file for cursor-up functionality
        history_dir = Path.home() / '.claude-cli'
        history_dir.mkdir(exist_ok=True)
        self.session = PromptSession(
            history=FileHistory(str(history_dir / 'command_history'))
        )

    def save_conversation(self, filename: str) -> None:
        """Save the current session's conversation history to a file"""
        try:
            filepath = Path(filename)
            if not filepath.suffix:
                filepath = filepath.with_suffix('.json')

            with open(filepath, 'w') as f:
                json.dump(self.history.session_history, f, indent=2)
            print(f"Conversation saved to {filepath}")
        except Exception as e:
            print(f"Error saving conversation: {e}")

    def load_conversation(self, filename: str) -> None:
        """Load conversation history from a file into the current session"""
        try:
            filepath = Path(filename)
            with open(filepath) as f:
                loaded_history = json.load(f)
            self.history.session_history = loaded_history
            print(f"Conversation loaded from {filepath}")
        except Exception as e:
            print(f"Error loading conversation: {e}")

    def interactive_mode(self):
        print("\nClaude CLI - Interactive Mode")
        print("Commands:")
        print("  !exit         Exit the program")
        print("  !clear        Clear current session history")
        print("  !run [n]      Run command block n (default: 1)")
        print("  !run all      Run all command blocks")
        print("  !run select   Choose command block interactively")
        print("  !share        Share last command output with Claude")
        print("  !save <file>  Save current session to file")
        print("  !load <file>  Load conversation from file")
        print("  Ctrl+C        Stop running command")
        print("  Ctrl+D        Exit the program")
        print("  Up/Down       Navigate command history")

        while True:
            try:
                user_input = self.session.prompt("\n> ").strip()

                if not user_input:
                    continue

                if not self.handle_command(user_input):
                    # Get previous messages for context
                    previous_messages = self.history.get_messages_for_api()

                    # Send message with conversation history
                    response = self.api.send_message(
                        user_input,
                        self.config.get_system_prompt(),
                        previous_messages
                    )
                    print(response)
                    self.history.add_interaction(user_input, response)

            except KeyboardInterrupt:
                print("\nUse !exit to quit")
            except EOFError:
                print("\nGoodbye!")
                sys.exit(0)
            except Exception as e:
                print(f"Error: {e}")

    def single_message_mode(self, message: str):
        previous_messages = self.history.get_messages_for_api()
        response = self.api.send_message(
            message,
            self.config.get_system_prompt(),
            previous_messages
        )
        print(response)

    def handle_command(self, cmd: str) -> bool:
        try:
            if cmd == "!exit":
                sys.exit(0)
            elif cmd == "!clear":
                self.history.clear_history()
                self.command_outputs.clear()  # Clear stored outputs
                print("History cleared")
                return True
            elif cmd.startswith("!save "):
                filename = cmd.split(maxsplit=1)[1]
                self.save_conversation(filename)
                return True
            elif cmd.startswith("!load "):
                filename = cmd.split(maxsplit=1)[1]
                self.load_conversation(filename)
                return True

            elif cmd.startswith("!run"):
                parts = cmd.split(maxsplit=1)
                block_num = parts[1] if len(parts) > 1 else "1"

                commands = self.history.get_last_commands()
                if not commands:
                    print("No commands found in last response")
                    return True

                self.command_outputs.clear()

                if block_num == "all":
                    for i, cmd in enumerate(commands, 1):
                        print(f"\nExecuting block {i}:")
                        try:
                            # Run with capture_output=True to store for !share
                            output = self.executor.execute_command(cmd, capture_output=True)
                            if output:
                                self.command_outputs.append((i, output))
                        except KeyboardInterrupt:
                            print("\nExecution stopped by user")
                            break

                elif block_num == "select":
                    print("\nAvailable commands:")
                    for i, cmd in enumerate(commands, 1):
                        print(f"\nBlock {i}:\n{cmd}")
                    try:
                        selection = int(input("\nSelect block number: "))
                        if 1 <= selection <= len(commands):
                            output = self.executor.execute_command(commands[selection-1], capture_output=True)
                            if output:
                                self.command_outputs.append((selection, output))
                    except ValueError:
                        print("Invalid selection")
                    except KeyboardInterrupt:
                        print("\nSelection cancelled")
                else:
                    try:
                        idx = int(block_num) - 1
                        if 0 <= idx < len(commands):
                            output = self.executor.execute_command(commands[idx], capture_output=True)
                            if output:
                                self.command_outputs.append((idx + 1, output))
                        else:
                            print(f"Block number {block_num} out of range")
                    except ValueError:
                        print(f"Invalid block number: {block_num}")
                return True

            elif cmd == "!share":
                if self.command_outputs:
                    # Format multiple outputs with block numbers
                    formatted_outputs = "\n\n".join([
                        f"Output from block {block_num}:\n```\n{output}\n```"
                        for block_num, output in self.command_outputs
                    ])

                    response = self.api.send_message(
                        f"Here is the output from my commands:\n\n{formatted_outputs}",
                        self.config.get_system_prompt()
                    )
                    print(response)
                    self.history.add_interaction(
                        f"Command outputs:\n{formatted_outputs}",
                        response
                    )
                else:
                    print("No command outputs to share")
                return True

            return False

        except KeyboardInterrupt:
            print("\nCommand cancelled")
            return True

def main():
    cli = CLI()

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
