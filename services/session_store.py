"""
Session Store Service for managing conversation state
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
from models.schemas import ChatSession, ChatMessage, UserProfile
from langchain.schema import HumanMessage, AIMessage, BaseMessage
import threading
import uuid

class SessionStore:
    """In-memory session store for managing conversation state"""
    
    def __init__(self, session_timeout_minutes: int = 60):
        self.sessions: Dict[str, ChatSession] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self._lock = threading.Lock()
    
    def create_session(self) -> ChatSession:
        """Create a new chat session"""
        with self._lock:
            session = ChatSession()
            self.sessions[session.session_id] = session
            print(f"📝 Created new session: {session.session_id}")
            return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get session by ID, return None if not found or expired"""
        with self._lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            
            # Check if session has expired
            if datetime.now() - session.last_activity > self.session_timeout:
                print(f"🕐 Session {session_id} expired, removing...")
                del self.sessions[session_id]
                return None
            
            return session
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity timestamp"""
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id].last_activity = datetime.now()
                return True
            return False
    
    def add_message_to_session(self, session_id: str, message: ChatMessage) -> bool:
        """Add a message to session conversation history"""
        with self._lock:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            session.conversation_history.append(message)
            session.last_activity = datetime.now()
            
            print(f"💬 Added {message.role} message to session {session_id}: {message.content[:50]}...")
            return True
    
    def update_session_profile(self, session_id: str, user_profile: UserProfile) -> bool:
        """Update user profile for a session"""
        with self._lock:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            session.user_profile = user_profile
            session.last_activity = datetime.now()
            
            print(f"👤 Updated profile for session {session_id}: {user_profile.first_name} {user_profile.last_name}")
            return True
    
    def update_session_phase(self, session_id: str, phase: str) -> bool:
        """Update conversation phase for a session"""
        with self._lock:
            if session_id not in self.sessions:
                return False
            
            old_phase = self.sessions[session_id].current_phase
            self.sessions[session_id].current_phase = phase
            self.sessions[session_id].last_activity = datetime.now()
            
            print(f"🎯 Updated phase for session {session_id}: {old_phase} -> {phase}")
            return True
    
    def get_langchain_messages(self, session_id: str) -> List[BaseMessage]:
        """Convert session conversation history to LangChain message format"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        messages = []
        for msg in session.conversation_history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        
        return messages
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                print(f"🗑️ Deleted session: {session_id}")
                return True
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions and return count of removed sessions"""
        with self._lock:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                if current_time - session.last_activity > self.session_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
            
            if expired_sessions:
                print(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
            
            return len(expired_sessions)
    
    def get_session_count(self) -> int:
        """Get current number of active sessions"""
        with self._lock:
            return len(self.sessions)
    
    def get_all_session_ids(self) -> List[str]:
        """Get list of all active session IDs"""
        with self._lock:
            return list(self.sessions.keys())

# Global session store instance
session_store = SessionStore(session_timeout_minutes=60)