
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

# Load environment variables from .env file
load_dotenv()

# Load system prompt from file
def load_system_prompt():
    """Load the system prompt from system_prompt.txt file."""
    prompt_file = Path("system_prompt.txt")
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    else:
        # Fallback prompt if file doesn't exist
        return "You are a helpful AI assistant."

def load_example_prompts():
    """Load example prompts from example_prompts.txt; fallback to defaults."""
    prompts_file = Path("example_prompts.txt")
    if prompts_file.exists():
        lines = [ln.strip() for ln in prompts_file.read_text(encoding="utf-8").splitlines()]
        return [ln for ln in lines if ln and not ln.startswith("#")]
    return [
        "Tell me about a breaking news story happening in Times Square.",
        "What's the latest development in NYC's tech scene?",
        "Tell me about any upcoming events at Madison Square Garden",
    ]
# Page configuration
st.set_page_config(
    page_title="AI News Reporter",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Get API key from environment or user input
    env_api_key = os.getenv("OPENAI_API_KEY", "")
    
    # Only show input if API key is not in environment
    if env_api_key:
        st.success("✅ API key loaded from environment")
        api_key = env_api_key
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value="",
            help="Enter your OpenAI API key or set OPENAI_API_KEY in .env file"
        )
    
    if api_key:
        # Initialize agent if not already done or if API key changed
        if st.session_state.agent is None or st.session_state.agent.model.api_key != api_key:
            try:
                # Load system prompt from file
                system_prompt = load_system_prompt()
                
                st.session_state.agent = Agent(
                    model=OpenAIChat(id="gpt-4o", api_key=api_key),
                    instructions=system_prompt,
                    tools=[DuckDuckGoTools(backend="html")],
                    markdown=True,
                )
                st.success("✅ Agent initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing agent: {str(e)}")
    else:
        st.warning("⚠️ Please enter your OpenAI API key to start")
    
    st.divider()
    
    # Example prompts
    st.subheader("💡 Example Prompts")
    example_prompts = load_example_prompts()
    
    for prompt in example_prompts:
        if st.button(prompt, key=f"example_{prompt[:20]}", use_container_width=True):
            st.session_state.example_prompt = prompt
            st.rerun()

# Main content area
st.title("🗽 AI News Reporter")
st.markdown("Your AI News Buddy that can search the web for real-time news with a distinctive NYC personality!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle example prompt from sidebar
if "example_prompt" in st.session_state:
    prompt = st.session_state.example_prompt
    del st.session_state.example_prompt
    # Add to messages and process
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Chat input
if prompt := st.chat_input("Ask me about the latest news..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    if st.session_state.agent is None:
        with st.chat_message("assistant"):
            st.error("⚠️ Please enter your OpenAI API key in the sidebar to start chatting.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching for news..."):
                try:
                    # Get response from agent
                    response = st.session_state.agent.run(prompt, stream=False)
                    
                    # Display response
                    st.markdown(response.content)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

