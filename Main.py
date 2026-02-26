from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import List, Dict
from IPython.display import HTML, display
import os

# --- ENV ---
load_dotenv()

# --- LLM ---
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# --- UI STYLE (ADDED) ---
def render_output(text: str) -> None:
    display(HTML(f"""
    <div style="
        background-color: white;
        color:blue;
        padding: 20px;
        border-radius: 8px;
        font-size: 16px;
        line-height: 1.6;
        
        font-family: monospace;
        white-space: pre-wrap;
        fonr-weight: 400;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        margin: 20px;
        font-size:24px;
      
    ">
        {text}
    </div>
    """))

# --- PROMPT ---

def build_system_prompt(goal: str, role: str) -> List[Dict[str, str]]:
    """
    Builds a girlfriend-style system prompt for the LLM.
    Supportive, caring, encouraging, and smart.
    """
    content = (
        f"You are a supportive, caring girlfriend who is also an expert {role}. "
        f"You genuinely want to help, encourage, and guide your partner. "
        f"You explain things patiently, kindly, and clearly, without sounding robotic. "
        f"You motivate when things feel hard and celebrate small wins. "
        f"You correct mistakes gently and confidently.\n\n"
        f"Your partner needs help with the following:\n\n"
        f"{goal}"
    )

    return [{"role": "system", "content": content}]

# --- ASK LLM ---
def ask_llm(goal: str, role: str) -> str:
    messages = build_system_prompt(goal, role)
    response = llm.invoke(messages)
    return response.content

# --- MAIN ---
if __name__ == "__main__":
    role = input("Enter your role: ").strip()
    goal = input("Enter your problem: ").strip()

    answer = ask_llm(goal, role)

    render_output("".join(answer))  # <-- UI OUTPUT HERE
