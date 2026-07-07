import os
from dotenv import load_dotenv
load_dotenv()
import re
import time
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_MAPPING = {
    "deepseek": "deepseek-r1:1.5b",
    "qwen": "qwen2.5:3b",
    "tinyllama": "tinyllama",
    "gemma": "gemma:2b"
}

# -------------------------------------------------------------
# RUNNING API PROTOCOLS
# -------------------------------------------------------------
try:
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception:
    gemini_client = None

try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception:
    groq_client = None
# -------------------------------------------------------------


def ask_model(model: str, prompt: str):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False, "keep_alive": "0"},
            timeout=90
        )
        return response.json().get("response", "ERROR")
    except Exception as e:
        return f"Local connection error: {str(e)}"


def ask_gemini(prompt: str):
    """Routes queries to Gemini with automatic exponential backoff retry for 503 spikes."""
    if not gemini_client: 
        return "ERROR: Gemini API Key missing or client uninitialized."
    
    retries = 3
    delay = 2
    for attempt in range(retries):
        try:
            return gemini_client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=prompt
            ).text
        except Exception as e:
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                return "⚠️ Gemini is currently saturated. Running fallback architectures..."
            return f"Gemini connection fault: {str(e)}"


def ask_groq(prompt: str):
    """Routes to Meta Llama-3.3-70b-versatile via Groq."""
    if not groq_client: 
        return "ERROR: Groq API Key missing or client uninitialized."
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return completion.choices[0].message.content
    except Exception as e: 
        return f"Groq execution fault: {str(e)}"


def ask_gpt_oss(prompt: str):
    """Routes to OpenAI open-weight reasoning model via Groq."""
    if not groq_client:
        return "ERROR: Groq API Key missing or client uninitialized."
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq GPT-OSS execution fault: {str(e)}"


def execute_routing(model_id: str, prompt: str) -> str:
    """Helper function to cleanly route prompts to any specific model ID."""
    if model_id == "gemini":
        return ask_gemini(prompt)
    elif model_id in ["meta", "groq"]:
        return ask_groq(prompt)
    elif model_id == "gptoss":
        return ask_gpt_oss(prompt)
    elif model_id in MODEL_MAPPING:
        raw = ask_model(MODEL_MAPPING[model_id], prompt)
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    else:
        # Fallback default model
        raw = ask_model("qwen2.5:3b", prompt)
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


@app.get("/")
def home():
    return {"message": "Direct-API Prototype Matrix Active"}


# ==========================================
# CHAINED ROUTING PIPELINE (Precision X1)
# ==========================================
@app.get("/ask")
def ask(prompt: str, models: str = "meta,gptoss,deepseek"):
    total_start = time.time()
    
    selected_list = [m.strip() for m in models.split(",") if m.strip()]
    if not selected_list:
        selected_list = ["meta", "gptoss", "deepseek"]

    current_context = ""
    last_response = ""
    
    # Pass 1: Direct Initial Answer Generation
    first_id = selected_list[0]
    last_response = execute_routing(first_id, prompt)
    current_context += f"Initial Output ({first_id}):\n{last_response}\n\n"

    # Pass 2: Context Review & Iterative Optimization Passes
    for model_id in selected_list[1:-1]:
        opt_prompt = f"Optimize and structuralize this textual output response:\n\n{last_response}"
        last_response = execute_routing(model_id, opt_prompt)
        current_context += f"Refinement Pass ({model_id}):\n{last_response}\n\n"

    # Pass 3: Final Matrix Synthesis Formulation
    if len(selected_list) > 1:
        final_id = selected_list[-1]
        synthesis_prompt = f"Synthesize a complete final response using the context trace log. Return ONLY clean output.\n\nQuestion:\n{prompt}\n\nTrace History:\n{current_context}"
        final_answer = execute_routing(final_id, synthesis_prompt)
    else:
        final_answer = last_response

    return {
        "final_answer": final_answer, 
        "total_time": f"{round(time.time() - total_start, 2)}s"
    }


# ==========================================
# DIAGNOSTIC SIDE-BY-SIDE MATRIX (Precision Glass)
# ==========================================
@app.get("/glass")
def glass(prompt: str, models: str = "meta,gptoss,deepseek"):
    selected_list = [m.strip() for m in models.split(",") if m.strip()]
    output_payload = {}

    for m_id in selected_list:
        start_time = time.time()
        res = execute_routing(m_id, prompt)
        
        key_name = "tinyllama_review" if m_id == "tinyllama" else m_id
        output_payload[key_name] = {
            "response": res.strip(), 
            "time_taken": f"{round(time.time() - start_time, 2)}s"
        }

    return output_payload


# ==========================================
# ORCHESTRATED ROLE DISPATCH WITH METACOGNITION (Precision X2)
# ==========================================
@app.get("/x2")
def x2_pipeline(prompt: str, xrole: str = "meta", workers: str = "gptoss,deepseek"):
    total_start = time.time()
    
    # Clean up chosen worker models list (Up to 3 permitted by front-end guard)
    worker_list = [w.strip() for w in workers.split(",") if w.strip()]
    if not worker_list:
        worker_list = ["deepseek"]

    # 1. METACOGNITIVE ORCHESTRATION PHASE
    # Forces the master model to analyze its own constraints and match workers based on cognitive strengths
    orchestration_prompt = (
        f"You are the master coordinator model operating with high metacognition (the ability to monitor and evaluate your own thinking).\n\n"
        f"TASK:\n"
        f"Analyze the user's prompt, identify your own initial cognitive blind spots or assumptions about it, "
        f"and strategically delegate sub-tasks to these available worker models: {worker_list}.\n\n"
        f"Original User Prompt: \"{prompt}\"\n\n"
        f"Provide your response in this exact structural format:\n"
        f"1. COGNITIVE DECONSTRUCTION: What is hidden or complex about this prompt? What skills are needed?\n"
        f"2. WORKER ROLES & ASSIGNMENTS: Map each worker from {worker_list} to a precise cognitive role. "
        f"Explicitly tell each worker what perspective to take, what to look out for, and what errors to avoid."
    )
    
    orchestration_plan = execute_routing(xrole, orchestration_prompt)
    
    # 2. METACOGNITIVE ROLE EXECUTION PHASE
    # Forces each worker to reason through its task, check its own work, and flag its uncertainties
    worker_logs = ""
    for worker_id in worker_list:
        worker_prompt = (
            f"You are an expert agent operating in an orchestrated cluster. You must apply deep metacognition to your output.\n\n"
            f"Master Strategy Blueprint & Role Assignment:\n{orchestration_plan}\n\n"
            f"Original Target Query: \"{prompt}\"\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"1. Step back and evaluate the assignment given to you by the master coordinator.\n"
            f"2. Before delivering your answer, evaluate your own reasoning. Are there logic gaps, missing edge cases, or biased assumptions in your approach?\n"
            f"3. Output your response using this structure:\n"
            f"   - [Self-Correction/Thinking]: A short breakdown of your internal logic checks and assumptions corrected.\n"
            f"   - [Role Execution Output]: Your clean, highly optimized final response for your assigned role."
        )
        
        worker_output = execute_routing(worker_id, worker_prompt)
        worker_logs += f"--- Worker [{worker_id}] Metacognitive Output ---\n{worker_output}\n\n"

    # 3. METACOGNITIVE CONSOLIDATED SYNTHESIS PHASE
    # The master model cross-references worker insights, flags conflicting logic, and builds a verified final answer
    synthesis_prompt = (
        f"You are the master coordinator model. It is time for final synthesis. You must review the work done "
        f"by your cluster and look closely at their self-corrections.\n\n"
        f"Original User Prompt: \"{prompt}\"\n\n"
        f"Collected Worker Outputs & Cognitive Logs:\n{worker_logs}\n\n"
        f"FINAL SYNTHESIS INSTRUCTIONS:\n"
        f"1. Evaluate the worker outputs. Look for any conflicting reasoning or errors between them and resolve them logically.\n"
        f"2. Synthesize a single, definitive, highly professional final answer to the user's prompt.\n"
        f"3. Return ONLY the highly refined final answer. Do not include your own meta-commentary in the final text—just deliver the ultimate solution."
    )
    
    final_synthetic_answer = execute_routing(xrole, synthesis_prompt)
    
    return {
        "x2_response": final_synthetic_answer,
        "total_time": f"{round(time.time() - total_start, 2)}s"
    }