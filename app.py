from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
# Import the workflow and dependencies
from workflow.workflow import Workflow, WorkflowState
from models.schemas import *
from services.vector_service import VectorService
from services.session_store import session_store
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langchain_core.messages import convert_to_messages
import dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Medical Services ChatBot API",
    description="Microservice-based ChatBot for Medical Services Q&A",
    version="1.0.0"
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatService:
    """Simple chat service to handle basic functionality"""
    def __init__(self):
        pass
    
    def get_welcome_message(self) -> str:
        """Get welcome message"""
        return "שלום! אני הצ'אטבוט של קופות החולים. אשמח לעזור לך עם שאלות בנוגע לשירותי הקופה שלך. תוכל בבקשה לתת לי את שמך הפרטי ואת שם המשפחה שלך?"

# Initialize services
try:
    # Initialize Azure OpenAI
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version=dotenv.get_key(".env", "AZURE_OPENAI_API_VERSION"),
        temperature=0
    )
    
    # Initialize vector service
    vector_service = VectorService("indexes")
    
    # Initialize workflow
    workflow_instance = Workflow(llm=llm, vector_service=vector_service)
    compiled_workflow = workflow_instance.build_workflow()
    
    # Initialize simple chat service
    chat_service = ChatService()
    
    print("✅ All services initialized successfully")
    
except Exception as e:
    print(f"❌ Error initializing services: {e}")
    raise

def convert_chat_history_to_langchain_messages(chat_history: List[ChatMessage]) -> List[BaseMessage]:
    """Convert ChatMessage list to LangChain message format"""
    messages = []
    for msg in chat_history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
    return messages

def extract_response_from_workflow_result(final_state: Dict[str, Any]) -> tuple[str, UserProfile | None, str]:
    """Extract the response message, user profile, and phase from workflow result"""
    response_message = ""
    user_profile = None
    phase = "collection"  # Default phase
    
    # The final_state is structured as a dict where each key is a node name
    # We need to extract from the last update in the workflow
    
    # Look through all node updates to find the latest state
    for node_name, node_update in final_state.items():
        if not isinstance(node_update, dict):
            continue
            
        # Extract messages
        if "messages" in node_update and node_update["messages"]:
            messages = node_update["messages"]
            # Find the last AI message
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and hasattr(msg, 'content') and isinstance(msg.content, str):
                    response_message = msg.content
                    break
        
        # Extract user profile if available
        if "user_profile" in node_update and node_update["user_profile"]:
            user_profile = node_update["user_profile"]
            
        # Extract phase if available  
        if "phase" in node_update and node_update["phase"]:
            phase = node_update["phase"]
    return response_message, user_profile, phase

# --- Clean Output Functions for Terminal Display ---

def clean_print_message(message, indent=False):
    """Clean print a single message for terminal output"""
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)

def clean_print_messages(update, last_message=False):
    """Clean print messages from workflow updates - less verbose output"""
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        if "messages" in node_update:
            messages = convert_to_messages(node_update["messages"])
            if last_message:
                messages = messages[-1:]

            for m in messages:
                clean_print_message(m, indent=is_subgraph)
        print("\n")

# --- Debug Functions to Print Chatbot Flow (for debug endpoints only) ---

def pretty_print_message(message, indent=False):
    """Pretty print a single message"""
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)

def pretty_print_messages(update, last_message=False):
    """Pretty print messages from workflow updates"""
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        if "messages" in node_update:
            messages = convert_to_messages(node_update["messages"])
            if last_message:
                messages = messages[-1:]

            for m in messages:
                pretty_print_message(m, indent=is_subgraph)
        print("\n")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Medical Services ChatBot API is running"}

@app.get("/welcome")
async def get_welcome_message():
    """Get initial welcome message"""
    return {"message": chat_service.get_welcome_message()}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint - stateless conversation handling"""
    try:
        # Validate request
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Convert chat history to LangChain message format
        langchain_messages = convert_chat_history_to_langchain_messages(request.conversation_history)
        
        # Add the current user message
        langchain_messages.append(HumanMessage(content=request.message))
        
        # Prepare initial state for workflow
        initial_state: WorkflowState = {
            "messages": langchain_messages,
            "user_profile": request.user_profile,
            "phase": request.phase  # Use the phase from request
        }
        
        # Run the workflow
        final_state = None
        try:    
            for chunk in compiled_workflow.stream(
                initial_state,
                config={"recursion_limit": 50}
            ):
                # Use clean output for terminal display
                clean_print_messages(chunk, last_message=True)
                final_state = chunk
                
        except Exception as workflow_error:
            print(f"Workflow error: {workflow_error}")
            raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(workflow_error)}")
        
        if not final_state:
            raise HTTPException(status_code=500, detail="No response from workflow")
        
        # Extract response, user profile, and phase from workflow result
        response_message, updated_user_profile, workflow_phase = extract_response_from_workflow_result(final_state)
        
        if not response_message:
            response_message = "מצטער, אני לא הצלחתי לעבד את הבקשה שלך. אנא נסה שוב."
        
        # Check if confirmation is required (when user profile is extracted but we're still in onboarding)
        requires_confirmation = (updated_user_profile is not None and 
                               workflow_phase == "qa" and 
                               request.user_profile is None)
        
        return ChatResponse(
            message=response_message,
            user_profile=updated_user_profile,
            phase=workflow_phase,
            requires_confirmation=requires_confirmation
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# === STATEFUL CHAT ENDPOINTS ===

@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session():
    """Create a new chat session"""
    try:
        session = session_store.create_session()
        welcome_message = chat_service.get_welcome_message()
        
        # Add welcome message to session history
        welcome_chat_message = ChatMessage(role="assistant", content=welcome_message)
        session_store.add_message_to_session(session.session_id, welcome_chat_message)
        
        return SessionCreateResponse(
            session_id=session.session_id,
            message=welcome_message
        )
    except Exception as e:
        print(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")

@app.get("/sessions/{session_id}", response_model=SessionInfoResponse)
async def get_session(session_id: str):
    """Get session information"""
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    return SessionInfoResponse(
        session_id=session.session_id,
        conversation_history=session.conversation_history,
        user_profile=session.user_profile,
        phase=session.current_phase,
        created_at=session.created_at,
        last_activity=session.last_activity
    )

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    success = session_store.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session deleted successfully"}

@app.post("/chat/stateful", response_model=StatefulChatResponse)
async def stateful_chat(request: StatefulChatRequest):
    """Stateful chat endpoint - maintains conversation state server-side"""
    try:
        # Validate request
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Get or create session
        if request.session_id:
            session = session_store.get_session(request.session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found or expired")
        else:
            # Create new session if none provided
            session = session_store.create_session()
            # Add welcome message if this is a new session
            welcome_message = chat_service.get_welcome_message()
            welcome_chat_message = ChatMessage(role="assistant", content=welcome_message)
            session_store.add_message_to_session(session.session_id, welcome_chat_message)
        
        if request.debug:
            print(f"\n🔧 STATEFUL DEBUG MODE - Session: {session.session_id}")
            print(f"📊 Session state BEFORE: profile={'✅' if session.user_profile else '❌'}, phase={session.current_phase}")
            print(f"📋 Message count: {len(session.conversation_history)}")
        
        # Add user message to session
        user_message = ChatMessage(role="user", content=request.message)
        session_store.add_message_to_session(session.session_id, user_message)
        
        # Get conversation history in LangChain format
        langchain_messages = session_store.get_langchain_messages(session.session_id)
        
        # Determine initial phase for workflow - CRITICAL: Use current session phase
        initial_phase = session.current_phase if session.current_phase else "collection"
        
        if request.debug:
            print(f"🎯 Initial phase for workflow: {initial_phase}")
        
        # Prepare initial state for workflow
        initial_state: WorkflowState = {
            "messages": langchain_messages,
            "user_profile": session.user_profile,
            "phase": initial_phase  # CRITICAL: Pass the current phase
        }
        
        # Run the workflow
        final_state = None
        all_updates = []  # Store all updates to better extract final state
        
        try:
            if request.debug:
                print("=" * 80)
                print("🚀 Starting workflow execution...")
                
            for chunk in compiled_workflow.stream(
                initial_state,
                config={"recursion_limit": 50}
            ):
                if request.debug:
                    pretty_print_messages(chunk, last_message=True)
                else:
                    # Use clean output for terminal display when not in debug mode
                    clean_print_messages(chunk, last_message=True)
                all_updates.append(chunk)
                final_state = chunk
                
            if request.debug:
                print("=" * 80)
                print(f"✅ Stateful workflow execution completed")
                print(f"📊 Total workflow updates: {len(all_updates)}")
                
        except Exception as workflow_error:
            print(f"Stateful workflow error: {workflow_error}")
            raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(workflow_error)}")
        
        if not final_state:
            raise HTTPException(status_code=500, detail="No response from workflow")
        
        # Extract response, user profile, and phase from workflow result
        response_message, updated_user_profile, workflow_phase = extract_response_from_workflow_result(final_state)
        
        if not response_message:
            response_message = "מצטער, אני לא הצלחתי לעבד את הבקשה שלך. אנא נסה שוב."
        
        # CRITICAL: Update session state properly
        session_updated = False
        
        # Update user profile if extracted
        if updated_user_profile and updated_user_profile != session.user_profile:
            session_store.update_session_profile(session.session_id, updated_user_profile)
            session_updated = True
            if request.debug:
                print(f"👤 Updated session profile: {updated_user_profile}")
        
        # Update phase if changed
        if workflow_phase and workflow_phase != session.current_phase:
            session_store.update_session_phase(session.session_id, workflow_phase)
            session_updated = True
            if request.debug:
                print(f"🎯 Updated session phase: {session.current_phase} -> {workflow_phase}")
        
        # Add assistant response to session
        assistant_message = ChatMessage(role="assistant", content=response_message)
        session_store.add_message_to_session(session.session_id, assistant_message)
        
        # Get updated session for response
        updated_session = session_store.get_session(session.session_id)
        
        # Check if confirmation is required
        requires_confirmation = (updated_user_profile is not None and 
                               workflow_phase == "qa" and 
                               session.user_profile is None)
        
        if request.debug:
            print(f"\n📤 STATEFUL FINAL RESPONSE:")
            print(f"   Session ID: {session.session_id}")
            print(f"   Message: {response_message[:100]}...")
            print(f"   Workflow Phase: {workflow_phase}")
            print(f"   Session Phase: {updated_session.current_phase if updated_session else 'N/A'}")
            print(f"   User Profile: {'✅ Available' if updated_user_profile else '❌ None'}")
            print(f"   Requires Confirmation: {requires_confirmation}")
            print(f"   Session Updated: {session_updated}")
            print("=" * 80)
        
        return StatefulChatResponse(
            message=response_message,
            session_id=session.session_id,
            user_profile=updated_session.user_profile if updated_session else updated_user_profile,
            phase=updated_session.current_phase if updated_session else workflow_phase,
            requires_confirmation=requires_confirmation
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error in stateful chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/vector-store/stats")
async def get_vector_store_stats():
    """Get vector store statistics"""
    try:
        stats = vector_service.get_stats()
        return stats
    except Exception as e:
        print(f"Error getting vector store stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving vector store statistics")

@app.post("/debug/chat")
async def debug_chat(request: ChatRequest):
    """Debug chat endpoint that always enables debug mode and shows workflow flow"""
    # Force debug mode
    request.debug = True
    
    print(f"\n{'='*80}")
    print(f"🛠 DEBUG CHAT SESSION STARTED")
    print(f"{'='*80}")
    
    # Call the regular chat endpoint
    response = await chat(request)
    
    print(f"{'='*80}")
    print(f"🛠 DEBUG CHAT SESSION ENDED")
    print(f"{'='*80}\n")
    
    return response

@app.post("/debug/chat/stateful", response_model=StatefulChatResponse)
async def debug_stateful_chat(request: StatefulChatRequest):
    """Debug stateful chat endpoint that always enables debug mode"""
    # Force debug mode
    request.debug = True
    
    print(f"\n{'='*80}")
    print(f"🛠 DEBUG STATEFUL CHAT SESSION STARTED")
    print(f"{'='*80}")
    
    # Call the stateful chat endpoint
    response = await stateful_chat(request)
    
    print(f"{'='*80}")
    print(f"🛠 DEBUG STATEFUL CHAT SESSION ENDED")
    print(f"{'='*80}\n")
    
    return response

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )