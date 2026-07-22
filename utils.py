"""
Utility functions for the Legal AI Assistant.

This module contains helper functions for logging, configuration management,
file handling, and other utility operations.

Author: Legal AI Team
Date: 2024
"""

import os
import logging
import logging.handlers
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from functools import wraps
import time


class ConfigManager:
    """
    Manages application configuration from YAML and environment variables.
    
    This class loads configuration from config.yaml and allows
    environment variables to override settings.
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[Dict[str, Any]] = None
    
    def __new__(cls) -> 'ConfigManager':
        """Implement singleton pattern for ConfigManager."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize ConfigManager and load configuration."""
        if self._config is None:
            self._load_config()
    
    def _load_config(self) -> None:
        """
        Load configuration from config.yaml file.
        
        Raises:
            FileNotFoundError: If config.yaml is not found.
            yaml.YAMLError: If YAML file is malformed.
        """
        config_path = Path(__file__).parent / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                ConfigManager._config = yaml.safe_load(f)
            logging.info(f"Configuration loaded from {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'document_processing.chunk_size')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = ConfigManager._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary."""
        return ConfigManager._config.copy()


class LoggerSetup:
    """
    Sets up logging for the application.
    
    Configures both console and file logging with appropriate
    formatters and handlers.
    """
    
    @staticmethod
    def setup_logger(
        name: str,
        log_level: str = "INFO",
        log_file: Optional[str] = None
    ) -> logging.Logger:
        """
        Set up a logger with specified configuration.
        
        Args:
            name: Logger name (typically __name__)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file. If None, config.yaml is used
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Get config
        config = ConfigManager()
        log_config = config.get('logging', {})
        
        if log_file is None:
            log_file = log_config.get('file', './logs/app.log')
        
        log_format = log_config.get(
            'format',
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_config.get('max_file_size', 10485760),
            backupCount=log_config.get('backup_count', 5)
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        
        return logger


class DirectoryManager:
    """
    Manages creation and validation of required directories.
    """
    
    @staticmethod
    def create_required_directories() -> None:
        """
        Create all required directories for the application.
        
        Creates:
        - logs/
        - data/
        - chroma_db/
        - database/
        - assets/
        """
        directories = [
            "./logs",
            "./data",
            "./chroma_db",
            "./database",
            "./assets"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logging.info(f"Directory ensured: {directory}")


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry a function on exception with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_on_exception(max_retries=3, delay=1.0)
        def risky_function():
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logging.error(
                            f"All {max_retries} attempts failed for {func.__name__}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


def validate_file_path(file_path: str, file_extension: Optional[str] = None) -> bool:
    """
    Validate if a file path exists and has the correct extension.
    
    Args:
        file_path: Path to the file
        file_extension: Expected file extension (e.g., '.pdf')
        
    Returns:
        True if file is valid, False otherwise
    """
    path = Path(file_path)
    
    if not path.exists():
        logging.error(f"File not found: {file_path}")
        return False
    
    if not path.is_file():
        logging.error(f"Path is not a file: {file_path}")
        return False
    
    if file_extension and not path.suffix.lower() == file_extension.lower():
        logging.error(
            f"Invalid file extension. Expected {file_extension}, got {path.suffix}"
        )
        return False
    
    return True


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    return Path(file_path).stat().st_size / (1024 * 1024)


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format datetime to readable string.
    
    Args:
        timestamp: Datetime object. If None, uses current time
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def sanitize_input(user_input: str, max_length: int = 5000) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        user_input: Raw user input
        max_length: Maximum allowed input length
        
    Returns:
        Sanitized input string
        
    Raises:
        ValueError: If input exceeds max_length
    """
    if len(user_input) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length} characters")
    
    # Remove leading/trailing whitespace
    sanitized = user_input.strip()
    
    # Remove null bytes
    sanitized = sanitized.replace('\x00', '')
    
    return sanitized


# Initialize logging at module level
logger = LoggerSetup.setup_logger(__name__)


if __name__ == "__main__":
    # Test configuration loading
    config = ConfigManager()
    print("Configuration loaded successfully")
    print(f"Chunk size: {config.get('document_processing.chunk_size')}")
    
    # Create required directories
    DirectoryManager.create_required_directories()
    print("Directories created successfully")
