"""
Convenience script to run both backend and frontend services
Updated for stateful session management
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if .env exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("🔧 Please copy .env.template to .env and fill in your Azure OpenAI credentials")
        return False
    
    # Check if vector store exists
    vector_store_path = Path("indexes")
    if not vector_store_path.exists() or not list(vector_store_path.glob("*.bin")):
        print("❌ Vector store not found!")
        print("🔧 Please run: python scripts/build_index.py")
        return False
    
    # Check if services directory exists
    services_path = Path("services")
    if not services_path.exists():
        print("❌ Services directory not found!")
        print("🔧 Please ensure the services directory exists with session_store.py")
        return False
    
    print("✅ All requirements satisfied!")
    return True

def run_backend():
    """Run FastAPI backend"""
    print("🚀 Starting FastAPI backend on http://localhost:8000")
    print("📝 Backend features:")
    print("   - Stateful session management")
    print("   - LangGraph workflow execution")
    print("   - Vector search integration")
    print("   - Debug endpoints available")
    return subprocess.Popen([
        sys.executable, "app.py"
    ])

def run_frontend():
    """Run Gradio frontend"""
    print("🌐 Starting Gradio frontend on http://localhost:7860")
    print("💡 Frontend features:")
    print("   - Server-side session state")
    print("   - Real-time status monitoring")
    print("   - Debug mode available")
    print("   - Session info tracking")
    return subprocess.Popen([
        sys.executable, "gradio_ui.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def main():
    """Main function"""
    print("🏥 Medical Services ChatBot - Stateful Startup Script")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    processes = []
    
    try:
        # Start backend
        backend_process = run_backend()
        processes.append(backend_process)
        
        # Wait a moment for backend to start
        print("⏳ Waiting for backend to start...")
        time.sleep(3)
        
        # Start frontend
        frontend_process = run_frontend()
        processes.append(frontend_process)
        
        print("\n🎉 Both services are starting up!")
        print("=" * 60)
        print("📊 Backend API: http://localhost:8000")
        print("💬 Frontend UI: http://localhost:7860")
        print("🛠 API Docs: http://localhost:8000/docs")
        print("=" * 60)
        print("\n🔄 System Features:")
        print("   ✅ Stateful session management")
        print("   ✅ Persistent conversation state")
        print("   ✅ Phase-based workflow routing")
        print("   ✅ Real-time status monitoring")
        print("   ✅ Debug mode support")
        print("\n📋 API Endpoints:")
        print("   📝 POST /sessions - Create new session")
        print("   💬 POST /chat/stateful - Send message (stateful)")
        print("   📊 GET /sessions/{id} - Get session info")
        print("   🛠 POST /debug/chat/stateful - Debug chat")
        print("\nPress Ctrl+C to stop both services")
        
        # Wait for processes
        while True:
            # Check if any process has died
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    print(f"\n❌ Process {i} has stopped unexpectedly")
                    # Read any error output
                    output, error = process.communicate()
                    if output:
                        print(f"Output: {output}")
                    if error:
                        print(f"Error: {error}")
                    raise KeyboardInterrupt()
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        
        # Terminate all processes
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        print("✅ All services stopped")
        print("💾 Note: Session data is stored in memory and will be lost")

if __name__ == "__main__":
    main()