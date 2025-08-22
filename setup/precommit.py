#!/usr/bin/env python3
"""
Setup script for pre-commit hooks.
Run this script to install and configure pre-commit hooks for the project.
"""

import os
import subprocess
import sys


def run_command(command: str, description: str) -> bool:
    """Run a command and handle errors gracefully."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def main() -> None:
    """Main setup function."""
    print("🚀 Setting up pre-commit hooks for StoryEvals...")

    # Check if we're in a git repository
    if not os.path.exists(".git"):
        print("❌ Error: This directory is not a git repository.")
        print("Please run 'git init' first or navigate to the correct directory.")
        sys.exit(1)

    # Install pre-commit
    if not run_command("uv add --dev pre-commit", "Installing pre-commit"):
        print("❌ Failed to install pre-commit. Please install it manually:")
        print("uv add --dev pre-commit")
        sys.exit(1)

    # Install pre-commit hooks
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        print("❌ Failed to install pre-commit hooks.")
        sys.exit(1)

    # Install additional dependencies for hooks
    print("📦 Installing additional dependencies for hooks...")
    dev_deps = [
        "mypy>=1.8.0",
        "types-requests",
        "types-PyYAML",
    ]

    for dep in dev_deps:
        if not run_command(f"uv add --dev {dep}", f"Installing {dep}"):
            print(f"⚠️  Warning: Failed to install {dep}")

    # Run pre-commit on all files
    print("🔍 Running pre-commit on all files...")
    if run_command("pre-commit run --all-files", "Running pre-commit checks"):
        print("✅ All pre-commit checks passed!")
    else:
        print("⚠️  Some pre-commit checks failed. Please fix the issues and run again.")

    print("\n🎉 Pre-commit setup complete!")
    print("\n📋 Available commands:")
    print("  pre-commit run --all-files  # Run all hooks on all files")
    print("  pre-commit run              # Run all hooks on staged files")
    print("  pre-commit run <hook-id>    # Run a specific hook")
    print("  pre-commit autoupdate       # Update hook versions")
    print("\n💡 The hooks will now run automatically on every commit!")


if __name__ == "__main__":
    main()
