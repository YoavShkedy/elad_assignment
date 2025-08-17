from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

class UserProfile(BaseModel):
    first_name: str = Field(..., description="User's first name")
    last_name: str = Field(..., description="User's last name") 
    national_id: str = Field(..., description="9-digit national ID number")
    gender: str = Field(..., description="Gender (male|female / זכר|נקבה)")
    date_of_birth: str = Field(..., description="Date of birth in DD/MM/YYYY format")
    hmo: str = Field(..., description="HMO name (Clalit|Maccabi|Meuhedet / כללית|מכבי|מאוחדת)")
    insurance_tier: str = Field(..., description="Insurance membership tier (gold|silver|bronze / זהב|כסף|ארד)")

class FieldExtraction(BaseModel):
    """Schema for extracting individual field values"""
    field: str = Field(..., description="The field name being extracted")
    value: str = Field(..., description="The extracted value for the field")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    user_profile: Optional[UserProfile] = Field(None, description="User profile if available")
    conversation_history: List[ChatMessage] = Field(default_factory=list, description="Chat history")
    phase: str = Field(default="collection", description="Current phase: collection or qa")
    debug: bool = Field(default=False, description="Enable debug mode to see workflow flow")

class ChatResponse(BaseModel):
    message: str = Field(..., description="Assistant's response")
    user_profile: Optional[UserProfile] = Field(None, description="Updated user profile")
    phase: str = Field(..., description="Current phase")
    requires_confirmation: bool = Field(default=False, description="Whether user needs to confirm profile")

class RetrievalResult(BaseModel):
    content: str = Field(..., description="Retrieved document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    score: float = Field(..., description="Similarity score")

class WorkflowState(BaseModel):
    message: str
    user_profile: Optional[UserProfile] = None
    conversation_history: List[ChatMessage] = Field(default_factory=list)
    phase: str = "collection"
    retrieved_docs: List[RetrievalResult] = Field(default_factory=list)
    response: str = ""
    requires_confirmation: bool = False
    
    # Information collection specific fields
    collection_complete: bool = False
    extraction_attempted: bool = False
    extraction_complete: bool = False

# Stateful Session Management Schemas
class ChatSession(BaseModel):
    """Represents a conversation session with persistent state"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_history: List[ChatMessage] = Field(default_factory=list)
    user_profile: Optional[UserProfile] = None
    current_phase: str = Field(default="collection", description="Current conversation phase")
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    
class StatefulChatRequest(BaseModel):
    """Request for stateful chat - only requires message and optional session_id"""
    message: str = Field(..., description="User's message")
    session_id: Optional[str] = Field(None, description="Session ID - if None, creates new session")
    debug: bool = Field(default=False, description="Enable debug mode")
    
class StatefulChatResponse(BaseModel):
    """Response from stateful chat - includes session info"""
    message: str = Field(..., description="Assistant's response")
    session_id: str = Field(..., description="Session ID for this conversation")
    user_profile: Optional[UserProfile] = Field(None, description="Current user profile")
    phase: str = Field(..., description="Current phase")
    requires_confirmation: bool = Field(default=False, description="Whether user needs to confirm profile")

class SessionCreateResponse(BaseModel):
    """Response when creating a new session"""
    session_id: str = Field(..., description="New session ID")
    message: str = Field(..., description="Welcome message")
    
class SessionInfoResponse(BaseModel):
    """Response with session information"""
    session_id: str = Field(..., description="Session ID")
    conversation_history: List[ChatMessage] = Field(default_factory=list, description="Full conversation history")
    user_profile: Optional[UserProfile] = Field(None, description="Current user profile")
    phase: str = Field(..., description="Current phase")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")