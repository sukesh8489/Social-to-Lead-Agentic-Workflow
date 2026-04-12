import sys
import uuid
from langchain_core.messages import HumanMessage
from agent import graph

def main():
    print("=========================================================================")
    print("Welcome to the AutoStream Conversational Agent (Social-to-Lead CLI Demo)")
    print("Type 'quit' or 'exit' to stop the application.")
    print("=========================================================================\n")

    # LangGraph uses a thread_id inside the config to manage conversational memory
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Pre-configure state for the first iteration or allow graph to build memory as it runs
    # Since we didn't add memory saver to the graph in agent.py explicitly for brevity (MemorySaver is standard),
    # let's modify graph in agent.py to use MemorySaver or we can just pass the running list of messages back and forth.
    
    # Actually, let's keep the running list of messages in this loop manually to ensure memory 
    # without needing sqlite memory saver, OR simply pass it.
    messages = []
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
                
            messages.append(HumanMessage(content=user_input))
            
            # Invoke the graph
            result = graph.invoke({"messages": messages}, config)
            
            # Update local messages list so memory is passed next time
            messages = result["messages"]
            
            # Print the AI's last message
            ai_msg = messages[-1].content
            if isinstance(ai_msg, list):
                # Claude can sometimes return content as a list of dicts
                ai_text = "".join(part.get("text", "") for part in ai_msg if isinstance(part, dict) and "text" in part)
            else:
                ai_text = str(ai_msg)
                
            if ai_text:
                print(f"Agent: {ai_text}")
            
            # If the tool mock_lead_capture was called, it would have printed to console directly inside agent.py
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()
