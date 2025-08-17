"""
Gradio frontend for Medical Services ChatBot - Stateful Client
"""

import gradio as gr
import httpx
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"

class StatefulChatState:
    """Client-side state management for stateful sessions"""
    
    def __init__(self):
        self.session_id = None
        self.current_phase = "collection"
        self.user_profile = None
    
    def set_session_id(self, session_id: str):
        """Set session ID"""
        self.session_id = session_id
    
    def update_from_response(self, response_data: Dict[str, Any]):
        """Update state from API response"""
        if "session_id" in response_data:
            self.session_id = response_data["session_id"]
        if "phase" in response_data:
            self.current_phase = response_data["phase"]
        if "user_profile" in response_data:
            self.user_profile = response_data["user_profile"]
    
    def reset(self):
        """Reset state"""
        self.session_id = None
        self.current_phase = "collection"
        self.user_profile = None

# Global state instance
chat_state = StatefulChatState()

async def create_new_session() -> Tuple[str, str]:
    """Create a new session and return session_id and welcome message"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{API_BASE_URL}/sessions")
            if response.status_code == 200:
                data = response.json()
                chat_state.set_session_id(data["session_id"])
                return data["session_id"], data["message"]
            else:
                return "", "Error creating session"
    except Exception as e:
        print(f"Error creating session: {e}")
        return "", "Error creating session"

async def send_stateful_message(message: str, debug: bool = False) -> Tuple[str, str, bool, str]:
    """Send message to stateful API and return response, phase, requires_confirmation, session_id"""
    try:
        request_data = {
            "message": message,
            "session_id": chat_state.session_id,
            "debug": debug
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/chat/stateful",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Update client state
                chat_state.update_from_response(result)
                
                return (
                    result["message"], 
                    result["phase"], 
                    result.get("requires_confirmation", False),
                    result["session_id"]
                )
            else:
                error_msg = f"API Error: {response.status_code}"
                try:
                    error_detail = response.json().get("detail", "Unknown error")
                    error_msg += f" - {error_detail}"
                except:
                    pass
                return error_msg, chat_state.current_phase, False, chat_state.session_id or ""
                
    except httpx.TimeoutException:
        return "Request timed out. Please try again.", chat_state.current_phase, False, chat_state.session_id or ""
    except Exception as e:
        print(f"Error sending message to API: {e}")
        return f"Error: {str(e)}", chat_state.current_phase, False, chat_state.session_id or ""

async def get_session_info() -> Dict[str, Any]:
    """Get current session information"""
    if not chat_state.session_id:
        return {"error": "No active session"}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/sessions/{chat_state.session_id}")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Session error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Error getting session info: {e}"}

async def process_user_message(message: str, history: List[Tuple[str, str]], debug_mode: bool = False) -> Tuple[List[Tuple[str, str]], str]:
    """Process user message and return updated history"""
    if not message.strip():
        return history, ""
    
    # If no session, create one
    if not chat_state.session_id:
        session_id, welcome_msg = await create_new_session()
        if session_id:
            # Add welcome message to history if it's not already there
            if not history or history[-1][1] != welcome_msg:
                history.append(("", welcome_msg))
    
    # Send message to API
    response_message, new_phase, requires_confirmation, session_id = await send_stateful_message(message, debug_mode)
    
    # Update history
    history.append((message, response_message))
    
    return history, ""

def reset_conversation():
    """Reset the conversation"""
    chat_state.reset()
    return [], ""

def get_current_phase_info() -> str:
    """Get information about current phase"""
    if not chat_state.session_id:
        return "**Status:** No active session"
    
    if chat_state.current_phase == "collection":
        profile_status = "❌ Not collected" if not chat_state.user_profile else "✅ Collected"
        return f"**Phase:** Collection\n**Session:** {chat_state.session_id[:8]}...\n**Profile:** {profile_status}"
    elif chat_state.current_phase == "qa":
        if chat_state.user_profile:
            return f"**Phase:** Q&A\n**Session:** {chat_state.session_id[:8]}...\n**User:** {chat_state.user_profile.get('first_name', '')} {chat_state.user_profile.get('last_name', '')}\n**HMO:** {chat_state.user_profile.get('hmo', '')}"
        else:
            return f"**Phase:** Q&A\n**Session:** {chat_state.session_id[:8]}...\n**Profile:** ❌ Missing"
    else:
        return f"**Phase:** {chat_state.current_phase}\n**Session:** {chat_state.session_id[:8] if chat_state.session_id else 'None'}..."

async def get_api_status() -> str:
    """Check API status"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_BASE_URL}/")
            if response.status_code == 200:
                return "🟢 API Connected"
            else:
                return f"🔴 API Error: {response.status_code}"
    except Exception as e:
        return f"🔴 API Offline: {str(e)}"

async def get_vector_store_status() -> str:
    """Check vector store status"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_BASE_URL}/vector-store/stats")
            if response.status_code == 200:
                stats = response.json()
                if stats["status"] == "loaded":
                    return f"🟢 Vector Store: {stats['total_documents']} documents"
                else:
                    return "🔴 Vector Store: Not loaded"
            else:
                return f"🔴 Vector Store Error: {response.status_code}"
    except Exception as e:
        return f"🔴 Vector Store Offline: {str(e)}"

async def get_session_status() -> str:
    """Get session status information"""
    if not chat_state.session_id:
        return "🔴 No active session"
    
    session_info = await get_session_info()
    if "error" in session_info:
        return f"🔴 Session Error: {session_info['error']}"
    
    msg_count = len(session_info.get("conversation_history", []))
    phase = session_info.get("phase", "unknown")
    has_profile = "✅" if session_info.get("user_profile") else "❌"
    
    return f"🟢 Session Active: {msg_count} messages, Phase: {phase}, Profile: {has_profile}"

def create_gradio_interface():
    """Create and configure Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .chat-container { max-height: 600px; overflow-y: auto; }
    """
    
    with gr.Blocks(css=css, title="Medical Services ChatBot - Stateful") as demo:
        
        gr.Markdown("""
        # 🤖 הצ'אטבוט של קופות החולים
        
        ## **עוזר בינה מלאכותית מתקדם לשירותי הבריאות בישראל**
        
        #### ברוכים הבאים לצ'אטבוט החכם של שירותי הרפואה! המערכת שלנו מיועדת לעזור לכם עם שאלות ובקשות הקשורות לשירותי קופות החולים בישראל - כללית, מכבי ומאוחדת.
        """)
        
        # Main chat interface
        chatbot = gr.Chatbot(
            label="Chat with Medical Services Bot",
            height=500,
            elem_classes=["chat-container"]
        )
        
        msg_input = gr.Textbox(
            label="Your Message",
            placeholder="Type your message here...",
            lines=2
        )
        
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Reset Conversation", variant="secondary")

        
        # Event handlers
        async def handle_send(message, history):
            """Handle send button click"""
            result_history, empty_msg = await process_user_message(message, history, debug_mode=False)
            return result_history, empty_msg
        
        async def handle_clear():
            """Handle clear button"""
            reset_conversation()
            return [], ""
        
        # Wire up events
        send_btn.click(
            fn=handle_send,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
            show_progress="hidden"
        )
        
        msg_input.submit(
            fn=handle_send,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
            show_progress="hidden"
        )
        
        clear_btn.click(
            fn=handle_clear,
            outputs=[chatbot, msg_input],
            show_progress="hidden"
        )
        
        # Create initial session and load welcome message on startup
        async def load_welcome():
            if not chat_state.session_id:
                session_id, welcome_msg = await create_new_session()
                if session_id:
                    return [(None, welcome_msg)]
            return []
        
        demo.load(
            fn=load_welcome,
            outputs=[chatbot],
            show_progress="hidden"
        )
        

    
    return demo

def main():
    """Main function to run Gradio interface"""
    print("Starting Medical Services ChatBot Gradio Interface (Stateful)...")
    
    demo = create_gradio_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()