"""
Chat history database module.

This module handles storing and retrieving chat conversations
using SQLite database.

Author: Legal AI Team
Date: 2024
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ChatDatabase:
    """
    Manages chat history and conversation storage.
    
    Handles:
    - Storing user questions and assistant answers
    - Retrieving chat history
    - Managing conversation metadata
    """
    
    def __init__(self, db_path: str = "./database/chats.db"):
        """
        Initialize ChatDatabase.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Create chat tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Chat database tables initialized successfully")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def create_conversation(self, user_id: int, title: Optional[str] = None) -> int:
        """
        Create a new conversation.
        
        Args:
            user_id: User ID
            title: Conversation title (optional)
            
        Returns:
            Conversation ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if title is None:
                title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            cursor.execute("""
                INSERT INTO conversations (user_id, title)
                VALUES (?, ?)
            """, (user_id, title))
            
            conversation_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Conversation created: {conversation_id} for user: {user_id}")
            return conversation_id
        
        except sqlite3.Error as e:
            logger.error(f"Error creating conversation: {e}")
            raise
    
    def add_message(
        self,
        conversation_id: int,
        user_id: int,
        role: str,
        content: str,
        sources: Optional[str] = None
    ) -> int:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID
            role: Message role ('user' or 'assistant')
            content: Message content
            sources: Source documents (JSON string, optional)
            
        Returns:
            Message ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO messages (conversation_id, user_id, role, content, sources)
                VALUES (?, ?, ?, ?, ?)
            """, (conversation_id, user_id, role, content, sources))
            
            message_id = cursor.lastrowid
            
            # Update conversation updated_at
            cursor.execute("""
                UPDATE conversations SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (conversation_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Message added: {message_id} to conversation: {conversation_id}")
            return message_id
        
        except sqlite3.Error as e:
            logger.error(f"Error adding message: {e}")
            raise
    
    def get_conversation_history(self, conversation_id: int) -> List[Dict]:
        """
        Get all messages in a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of message dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, role, content, sources, timestamp
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
            """, (conversation_id,))
            
            messages = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return messages
        
        except sqlite3.Error as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    def get_user_conversations(self, user_id: int, limit: int = 50) -> List[Dict]:
        """
        Get all conversations for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of conversations to retrieve
            
        Returns:
            List of conversation dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, title, created_at, updated_at
                FROM conversations
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
            """, (user_id, limit))
            
            conversations = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return conversations
        
        except sqlite3.Error as e:
            logger.error(f"Error retrieving user conversations: {e}")
            return []
    
    def delete_conversation(self, conversation_id: int) -> Tuple[bool, str]:
        """
        Delete a conversation and its messages.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete messages first
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            
            # Delete conversation
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Conversation deleted: {conversation_id}")
            return True, "Conversation deleted successfully"
        
        except sqlite3.Error as e:
            logger.error(f"Error deleting conversation: {e}")
            return False, f"Error deleting conversation: {str(e)}"
    
    def search_messages(
        self,
        user_id: int,
        query: str,
        limit: int = 20
    ) -> List[Dict]:
        """
        Search through user's messages.
        
        Args:
            user_id: User ID
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching message dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            search_term = f"%{query}%"
            
            cursor.execute("""
                SELECT m.id, m.conversation_id, m.role, m.content, m.timestamp,
                       c.title as conversation_title
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.user_id = ? AND m.content LIKE ?
                ORDER BY m.timestamp DESC
                LIMIT ?
            """, (user_id, search_term, limit))
            
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return results
        
        except sqlite3.Error as e:
            logger.error(f"Error searching messages: {e}")
            return []
    
    def get_conversation_stats(self, user_id: int) -> Dict:
        """
        Get statistics about user's conversations.
        
        Args:
            user_id: User ID
            
        Returns:
            Statistics dictionary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total conversations
            cursor.execute(
                "SELECT COUNT(*) FROM conversations WHERE user_id = ?",
                (user_id,)
            )
            total_conversations = cursor.fetchone()[0]
            
            # Total messages
            cursor.execute("""
                SELECT COUNT(*) FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.user_id = ?
            """, (user_id,))
            total_messages = cursor.fetchone()[0]
            
            # Messages per role
            cursor.execute("""
                SELECT role, COUNT(*) as count FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.user_id = ?
                GROUP BY role
            """, (user_id,))
            role_stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            conn.close()
            
            return {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "role_stats": role_stats
            }
        
        except sqlite3.Error as e:
            logger.error(f"Error getting conversation stats: {e}")
            return {}
    
    def export_conversation(self, conversation_id: int) -> Optional[Dict]:
        """
        Export a complete conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation data dictionary or None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get conversation info
            cursor.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            conv_row = cursor.fetchone()
            
            if not conv_row:
                return None
            
            # Get messages
            messages = self.get_conversation_history(conversation_id)
            
            conn.close()
            
            return {
                "conversation": dict(conv_row),
                "messages": messages
            }
        
        except sqlite3.Error as e:
            logger.error(f"Error exporting conversation: {e}")
            return None


if __name__ == "__main__":
    # Test chat database
    chat_db = ChatDatabase()
    
    # Create conversation
    conv_id = chat_db.create_conversation(1, "Test Conversation")
    print(f"Created conversation: {conv_id}")
    
    # Add messages
    chat_db.add_message(conv_id, 1, "user", "What is IPC?")
    chat_db.add_message(conv_id, 1, "assistant", "IPC is the Indian Penal Code...")
    
    # Get history
    history = chat_db.get_conversation_history(conv_id)
    print(f"History: {history}")
