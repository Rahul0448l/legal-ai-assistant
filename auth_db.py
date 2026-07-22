"""
Authentication and user database management module.

This module handles user registration, login, password hashing,
and user database operations using SQLite and bcrypt.

Author: Legal AI Team
Date: 2024
"""

import sqlite3
import bcrypt
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AuthDatabase:
    """
    Manages user authentication and database operations.
    
    Handles:
    - User registration
    - Login authentication
    - Password hashing with bcrypt
    - User profile management
    """
    
    def __init__(self, db_path: str = "./database/users.db"):
        """
        Initialize AuthDatabase.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Create users table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Users table initialized successfully")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Plain text password to verify
            password_hash: Hashed password to compare against
            
        Returns:
            True if password matches, False otherwise
        """
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def register_user(self, username: str, email: str, password: str) -> Tuple[bool, str]:
        """
        Register a new user.
        
        Args:
            username: Desired username
            email: User email address
            password: User password
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Validate input
            if not username or len(username) < 3:
                return False, "Username must be at least 3 characters long"
            
            if not email or '@' not in email:
                return False, "Invalid email address"
            
            if not password or len(password) < 6:
                return False, "Password must be at least 6 characters long"
            
            # Hash password
            password_hash = self.hash_password(password)
            
            # Insert user
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            """, (username, email, password_hash))
            
            conn.commit()
            conn.close()
            
            logger.info(f"User registered successfully: {username}")
            return True, "User registered successfully. Please login."
        
        except sqlite3.IntegrityError:
            logger.warning(f"Registration failed - username or email already exists: {username}")
            return False, "Username or email already exists. Please try another."
        
        except sqlite3.Error as e:
            logger.error(f"Database error during registration: {e}")
            return False, f"Database error: {str(e)}"
    
    def login_user(self, username: str, password: str) -> Tuple[bool, str, Optional[int]]:
        """
        Authenticate user login.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Tuple of (success: bool, message: str, user_id: Optional[int])
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, password_hash FROM users WHERE username = ?
            """, (username,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result is None:
                logger.warning(f"Login failed - user not found: {username}")
                return False, "Invalid username or password", None
            
            user_id, password_hash = result
            
            if self.verify_password(password, password_hash):
                logger.info(f"User logged in successfully: {username}")
                return True, "Login successful", user_id
            else:
                logger.warning(f"Login failed - incorrect password: {username}")
                return False, "Invalid username or password", None
        
        except sqlite3.Error as e:
            logger.error(f"Database error during login: {e}")
            return False, f"Database error: {str(e)}", None
    
    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        """
        Retrieve user information by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User dictionary or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, username, email, created_at FROM users WHERE id = ?
            """, (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return dict(result)
            return None
        
        except sqlite3.Error as e:
            logger.error(f"Error fetching user: {e}")
            return None
    
    def update_password(self, user_id: int, new_password: str) -> Tuple[bool, str]:
        """
        Update user password.
        
        Args:
            user_id: User ID
            new_password: New password
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if not new_password or len(new_password) < 6:
                return False, "Password must be at least 6 characters long"
            
            password_hash = self.hash_password(new_password)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE users SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (password_hash, user_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Password updated for user ID: {user_id}")
            return True, "Password updated successfully"
        
        except sqlite3.Error as e:
            logger.error(f"Error updating password: {e}")
            return False, f"Database error: {str(e)}"
    
    def delete_user(self, user_id: int) -> Tuple[bool, str]:
        """
        Delete user account.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"User deleted: {user_id}")
            return True, "Account deleted successfully"
        
        except sqlite3.Error as e:
            logger.error(f"Error deleting user: {e}")
            return False, f"Database error: {str(e)}"
    
    def get_all_users_count(self) -> int:
        """
        Get total number of registered users.
        
        Returns:
            Number of users
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM users")
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
        
        except sqlite3.Error as e:
            logger.error(f"Error getting user count: {e}")
            return 0


if __name__ == "__main__":
    # Test authentication module
    auth = AuthDatabase()
    
    # Test registration
    success, msg = auth.register_user("testuser", "test@example.com", "password123")
    print(f"Registration: {msg}")
    
    # Test login
    success, msg, user_id = auth.login_user("testuser", "password123")
    print(f"Login: {msg}, User ID: {user_id}")
