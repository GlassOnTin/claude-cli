#!/usr/bin/env python3

import os
import sys
import json
import tempfile
import subprocess
import requests
from typing import Optional, Tuple, List
from pathlib import Path

class Config:
    def __init__(self):
        self.check_dependencies()
        self.check_environment()
        self.history_file = self.init_history()

    @staticmethod
    def check_dependencies() -> None:
        # Python's standard library includes json, so no additional check needed
        pass

    @staticmethod
    def check_environment() -> None:
        if not os.getenv('ANTHROPIC_API_KEY'):
            print("Error: ANTHROPIC_API_KEY environment variable is not set")
            sys.exit(1)

    @staticmethod
    def init_history() -> str:
        history_file = tempfile.mktemp(prefix='claude_cli_history_', suffix='.txt')
        Path(history_file).touch()
        return history_file

    @staticmethod
    def get_system_prompt() -> str:
        return f"""You are a Linux shell assistant. I will help users write and understand shell commands and scripts.

Key points:
- When providing commands, always use ```bash code blocks
- Each command block should be self-contained and executable
- Explain what commands do before or after the code blocks
- Multiple command blocks are fine - users can select which to run

Current context:
Directory: {os.getcwd()}
Shell: {os.getenv('SHELL')}

Usage:
- `!run` or `!run 1` - Run first command
- `!run 2` etc - Run specific command block
- `!run select` - Choose from available commands
- `!share` - Share last command output with me for analysis"""

class API:
    def __init__(self, config: Config):
        self.config = config
        self.api_key = os.getenv('ANTHROPIC_API_KEY')

    def send_message(self, message: str) -> str:
        history = ""
        if os.path.exists(self.config.history_file):
            with open(self.config.history_file, 'r') as f:
                history = f.read()

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
                "system": self.config.get_system_prompt(),
                "messages": [{
                    "role": "user",
                    "content": f"Context: Working in bash shell on Ubuntu Linux.\n\nConversation history:\n{history}\n\nCurrent request: {message}"
                }]
            }
        )

        response.raise_for_status()
        return response.json()['content'][0]['text'].strip()

class History:
    def __init__(self, history_file: str):
        self.history_file = history_file

    def get_all_commands(self) -> List[Tuple[int, str]]:
        if not os.path.exists(self.history_file):
            return []

        with open(self.history_file, 'r') as f:
            content = f.read()

        blocks = []
        current_block = []
        block_num = 0
        in_block = False

        for line in reversed(content.split('\n')):
            if line.startswith('Assistant:'):
                break
            elif line == '```bash':
                in_block = True
                block_num += 1
                current_block = []
            elif line == '```' and in_block:
                if current_block:
                    blocks.append((block_num, '\n'.join(reversed(current_block)).strip()))
                in_block = False
            elif in_block:
                current_block.append(line)

    def get_last_command(self, block_number: Optional[str] = "1") -> str:
        blocks = self.get_all_commands()

        if not blocks:
            print("No executable commands found in last response")
            return ""

        if block_number == "select":
            print("Available command blocks:")
            for num, block in blocks:
                print(f"\nBlock {num}:\n{block}\n")

            while True:
                try:
                    selection = int(input(f"Select block number (1-{len(blocks)}): "))
                    if 1 <= selection <= len(blocks):
                        block_number = str(selection)
                        break
                except ValueError:
                    pass
                print(f"Please enter a number between 1 and {len(blocks)}")

        try:
            block_num = int(block_number)
            if 1 <= block_num <= len(blocks):
                return blocks[block_num - 1][1]
            else:
                print(f"Error: Block number {block_num} out of range")
                return ""
        except ValueError:
            print(f"Error: Invalid block number format: {block_number}")
            return ""

    def add_to_history(self, message: str, response: str) -> None:
        with open(self.history_file, 'a') as f:
            f.write(f"User: {message}\nAssistant: {response}\n")

    def clear_history(self) -> None:
        with open(self.history_file, 'w') as f:
            f.write("")

class Executor:
    @staticmethod
    def execute_command(cmd: str, capture_output: bool = False) -> Optional[str]:
        if not cmd:
            print("No command provided. Usage: !run <command>")
            return None

        print(f"DEBUG: Executing command:\n{cmd}")  # Add debug output

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write(cmd)
            tmp_path = tmp.name

        os.chmod(tmp_path, 0o755)

        print("Executing command...")
        print("---")

        try:
            env = os.environ.copy()
            env.pop('ANTHROPIC_API_KEY', None)

            result = subprocess.run(
                ['bash', tmp_path],
                capture_output=capture_output,
                text=True,
                env=env
            )

            if capture_output:
                output = result.stdout + result.stderr
                print(output)
                return output

            if result.returncode != 0:
                print(f"Command failed with exit code {result.returncode}")
                return None

        finally:
            os.unlink(tmp_path)
            print("---")

    def execute_all_commands(self, history: History, capture_output: bool = False) -> Optional[str]:
        blocks = history.get_all_commands()
        if not blocks:
            print("No commands found to execute")
            return None

        all_output = []
        for block_num, cmd in blocks:
            print(f"Executing block {block_num}...")
            print("---")
            output = self.execute_command(cmd, capture_output)
            if capture_output and output:
                all_output.append(f"Block {block_num} output:\n{output}")
            print("---")

        return '\n\n'.join(all_output) if capture_output else None

class CLI:
    def __init__(self):
        self.config = Config()
        self.api = API(self.config)
        self.history = History(self.config.history_file)
        self.executor = Executor()
        self.last_output = ""

    def interactive_mode(self):
        print("? Claude CLI - Interactive Mode")
        print("Type '!exit' to quit, '!clear' to clear history")
        print("Type '!run [n]' to execute command block n")
        print("Type '!run all' to execute all command blocks")
        print("Type '!run select' to choose a command block")
        print("Type '!share' to share last command output with Claude")

        while True:
            try:
                input_text = input("> ")

                if input_text == "!exit":
                    os.unlink(self.config.history_file)
                    sys.exit(0)

                elif input_text == "!clear":
                    self.history.clear_history()
                    print("Conversation history cleared")

                elif input_text == "!share":
                    if self.last_output:
                        response = self.api.send_message(
                            f"Here is the output from my last command(s): \n```\n{self.last_output}\n```"
                        )
                        print(response)
                        self.history.add_to_history(
                            f"Command output: \n```\n{self.last_output}\n```",
                            response
                        )
                    else:
                        print("No command output to share")

                elif input_text.startswith("!run"):
                    parts = input_text.split(maxsplit=1)
                    block_spec = parts[1] if len(parts) > 1 else "1"

                    if block_spec == "all":
                        self.last_output = self.executor.execute_all_commands(
                            self.history,
                            capture_output=True
                        )
                    else:
                        cmd = self.history.get_last_command(block_spec)
                        if cmd:
                            self.last_output = self.executor.execute_command(cmd, capture_output=True)

                else:
                    self.last_output = ""
                    response = self.api.send_message(input_text)
                    print(response)
                    self.history.add_to_history(input_text, response)

            except KeyboardInterrupt:
                print("\nUse !exit to quit")
            except Exception as e:
                print(f"Error: {e}")

    def show_help(self):
        print("Usage: claude-cli [options] [message]")
        print("Options:")
        print("  -h, --help    Show this help message")
        print("  No arguments  Enter interactive mode")
        sys.exit(0)

def main():
    cli = CLI()

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            cli.show_help()
        else:
            response = cli.api.send_message(sys.argv[1])
            print(response)
            cli.history.add_to_history(sys.argv[1], response)
    else:
        cli.interactive_mode()

if __name__ == "__main__":
    main()
