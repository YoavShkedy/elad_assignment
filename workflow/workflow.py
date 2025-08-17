import sys
import os
# Add the parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import List, Optional
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from models.schemas import UserProfile
from services.vector_service import VectorService

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import convert_to_messages
from services.agent_tools import Tools
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from typing import Annotated, TypedDict
import json

# Load environment variables
load_dotenv()

# Define the state for the workflow
class WorkflowState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_profile: Optional[UserProfile]
    phase: Optional[str]  # Track current phase: "collection" or "qa"

class Workflow:
    """
    Class that defines the workflow of the chatbot.
    """
    def __init__(self, llm: AzureChatOpenAI, vector_service: VectorService):
        self.llm = llm
        self.vector_service = vector_service
        self.tools = Tools(vector_service=self.vector_service, llm=self.llm)
        self.extraction_tool_node = ToolNode([self.tools.extract_user_info])
        self.search_tool_node = ToolNode([self.tools.search_info])
        self.agents = self.create_agents()


    def _load_prompt_from_file(self, filename: str) -> str:
        """Load prompt content from a file"""
        # Get the directory of this script and go up one level to find prompts
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(os.path.dirname(script_dir), "prompts")
        prompt_path = os.path.join(prompts_dir, filename)
        try:
            with open(prompt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    def create_agents(self):
        """
        Create the agents for the workflow.
        """
        # --- Create info collection agent ---

        # Load the prompt from the file
        info_collection_prompt_content = self._load_prompt_from_file("info_collection.txt")
        
        # Create collector agent with extraction tool
        collector_llm = self.llm.bind_tools([self.tools.extract_user_info])

        # Get conversation history and concatenate the prompt
        def get_messages_info(messages):
            return [SystemMessage(content=info_collection_prompt_content)] + messages

        # Collector agent that uses the extraction tool
        def collector_agent(state):
            messages = get_messages_info(state["messages"])
            response = collector_llm.invoke(messages)
            return {
                "messages": [response],
                "user_profile": state.get("user_profile"),
                "phase": state.get("phase")
            }

        # --- Create QA agent ---

        # Load the prompt from the file
        qa_prompt_content = self._load_prompt_from_file("qa.txt")
        
        # Create QA agent with search tool
        qa_llm = self.llm.bind_tools([self.tools.search_info])

        # Get conversation history and concatenate the prompt
        def get_messages_qa(messages):
            return [SystemMessage(content=qa_prompt_content)] + messages

        # QA agent that uses the search tool
        def qa_agent(state):
            messages = get_messages_qa(state["messages"])
            response = qa_llm.invoke(messages)
            return {
                "messages": [response],
                "user_profile": state.get("user_profile"),
                "phase": state.get("phase")
            }

        return {
            "collector_agent": collector_agent,
            "qa_agent": qa_agent
        }

    def build_workflow(self):
        """
        Build the workflow of the chatbot.
        """

        def route_after_collector(state: WorkflowState):
            """Route after collector agent - check if extraction tool was called"""
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # Last message is an AI message to the extraction tool
                return "handle_extraction_tool"
            elif not isinstance(last_message, HumanMessage):
                # Last message is an AI message
                return END
            # Last message is a human message
            return "collector_agent"

        def route_after_qa(state: WorkflowState):
            """Route after QA agent - check if search tool was called"""
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # Last message is an AI message to the search tool
                return "handle_search_tool"
            elif not isinstance(last_message, HumanMessage):
                # Last message is an AI message
                return END
            # Last message is a human message
            return "qa_agent"

        # Create the workflow graph
        workflow = StateGraph(WorkflowState)
    
        # Add entry point node for dynamic routing
        @workflow.add_node
        def entry_point(state: WorkflowState):
            """Entry point that routes based on current phase"""
            # If phase is "qa", we've already collected info and should continue with QA
            # Otherwise, start with collection
            return {"messages": []}  # No new messages, just routing
    
        # Add nodes
        workflow.add_node("collector_agent", self.agents["collector_agent"])
        workflow.add_node("qa_agent", self.agents["qa_agent"])
        
        # Add the tool message handler node for collector
        @workflow.add_node
        def handle_extraction_tool(state: WorkflowState):
            """Handle tool call and extract user profile from result"""
            # Use ToolNode to run tool and get ToolMessage
            result = self.extraction_tool_node.invoke(state)
            tool_messages = result["messages"]
            
            # Parse content of the ToolMessage to extract user profile
            try:
                content = tool_messages[0].content
                user_profile_data = json.loads(content)
                user_profile = UserProfile(**user_profile_data)
            except Exception as e:
                print(f"Failed to parse tool result into UserProfile: {e}")
                user_profile = None

            return {
                "messages": tool_messages,
                "user_profile": user_profile,
                "phase": "qa"  # Move to QA phase after profile extraction
            }

        # Add the tool message handler node for QA
        @workflow.add_node
        def handle_search_tool(state: WorkflowState):
            """Handle tool calls from QA agent"""
            # Use ToolNode to run tool and get ToolMessage
            result = self.search_tool_node.invoke(state)
            
            return {
                "messages": result["messages"]
            }

        # Start at the entry point instead of collector agent
        workflow.add_edge(START, "entry_point")
        
        # Add routing from entry point based on phase
        def route_from_entry(state: WorkflowState):
            """Route from entry point based on current phase"""
            phase = state.get("phase") or "collection"
            print(f"Phase: {state.get('phase')}")
            return "qa_agent" if phase == "qa" else "collector_agent"
        
        workflow.add_conditional_edges(
            "entry_point",
            route_from_entry,
            ["collector_agent", "qa_agent"]
        )
        
        # Add conditional edges from collector agent
        workflow.add_conditional_edges(
            "collector_agent",
            route_after_collector,
            ["handle_extraction_tool", END]
        )
        
        # Add edge from extraction tool to QA agent
        workflow.add_edge("handle_extraction_tool", "qa_agent")
        
        # Add conditional edges from QA agent
        workflow.add_conditional_edges(
            "qa_agent",
            route_after_qa,
            ["handle_search_tool", END]
        )
        
        # Add edge from QA tool handler back to QA agent
        workflow.add_edge("handle_search_tool", "qa_agent")
        
        # Compile and return the workflow
        return workflow.compile()

    def pretty_print_message(self, message, indent=False):
        """Pretty print a single message"""
        pretty_message = message.pretty_repr(html=True)
        if not indent:
            print(pretty_message)
            return

        indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
        print(indented)


    def pretty_print_messages(self, update, last_message=False):
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

            messages = convert_to_messages(node_update["messages"])
            if last_message:
                messages = messages[-1:]

            for m in messages:
                self.pretty_print_message(m, indent=is_subgraph)
            print("\n")
    
if __name__ == "__main__":
    
    # Load environment variables
    load_dotenv()

    # Initialize the workflow
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-12-01-preview",
        temperature=0
    )

    # Initialize the vector service
    vector_service = VectorService()

    # Initialize the workflow
    workflow = Workflow(llm=llm, vector_service=vector_service)

    # Build the workflow
    compiled_workflow = workflow.build_workflow()

    # Save the workflow graph to a file
    with open("workflow.png", "wb") as f:
        f.write(compiled_workflow.get_graph().draw_mermaid_png())
    
    # Initialize conversation history
    conversation_messages = []
    current_user_profile = None
    current_phase = "collection"

    # Print chat history like a user would see it
    while True:
        try:
            user_input = input("\n👤 You: ")
            if not user_input.strip():
                continue

            if user_input.lower() == "exit":
                print("\n\n👋 Thank you and goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Thank you and goodbye!")
            break

        # Add user message to conversation history
        conversation_messages.append(HumanMessage(content=user_input))

        # Run the multi-agent bot with full conversation history
        try:
            final_state = None
            for chunk in compiled_workflow.stream(
                {
                    "messages": conversation_messages,  # Pass full history
                    "user_profile": current_user_profile,  # Preserve user profile
                    "phase": current_phase  # Preserve current phase
                },
                config={"recursion_limit": 50}
            ):
                workflow.pretty_print_messages(chunk, last_message=True)
                # Keep track of the final state
                final_state = chunk
            
            # Update conversation history with agent responses and preserve state
            if final_state:
                for node_name, node_update in final_state.items():
                    if "messages" in node_update:
                        # Add the new agent messages to our history
                        new_messages = node_update["messages"]
                        conversation_messages.extend(new_messages)
                    
                    # Update preserved state from final state
                    if "user_profile" in node_update and node_update["user_profile"]:
                        current_user_profile = node_update["user_profile"]
                    if "phase" in node_update and node_update["phase"]:
                        current_phase = node_update["phase"]
        
        except Exception as e:
            print(f"❌ Error: {e}")