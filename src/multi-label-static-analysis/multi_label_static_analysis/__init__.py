"""Multi-label static malware analysis using EMBER2024."""


def main() -> None:
    """Compatibility entry point for the original project command."""
    from .cli import main as cli_main
    cli_main()
