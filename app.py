import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
import time

# Page configuration
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #0d6efd;
        color: white;
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 6px solid #084298;
    }
    .bot-message {
        background-color: #e9ecef;
        color: #212529;
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 6px solid #198754;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the model and tokenizer with caching"""
    with st.spinner("Loading medical chatbot model... This may take a few minutes."):
        try:
            PEFT_MODEL = "TestCase1/falcon-7b-lora-chat-medical-bot"

            # Load configuration
            config = PeftConfig.from_pretrained(PEFT_MODEL)

            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                return_dict=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
            tokenizer.pad_token = tokenizer.eos_token

            # Load PEFT model
            model = PeftModel.from_pretrained(model, PEFT_MODEL)

            # Configure generation parameters
            generation_config = model.generation_config
            generation_config.max_new_tokens = 100
            generation_config.temperature = 0.7
            generation_config.top_p = 0.7
            generation_config.num_return_sequences = 1
            generation_config.pad_token_id = tokenizer.eos_token_id
            generation_config.eos_token_id = tokenizer.eos_token_id
            generation_config.repetition_penalty = 1.5
            generation_config.no_repeat_ngram_size = 3
            generation_config.early_stopping = True

            return model, tokenizer, generation_config

        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None, None


def generate_response(model, tokenizer, generation_config, prompt, device="cuda:0"):
    """Generate response from the model"""
    try:
        # Format the prompt
        formatted_prompt = f": {prompt}?\n: "

        # Tokenize
        encoding = tokenizer(formatted_prompt, return_tensors="pt")

        # Move to device if available
        if torch.cuda.is_available() and device == "cuda:0":
            encoding = encoding.to(device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                generation_config=generation_config,
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (remove the prompt)
        response = response.replace(formatted_prompt, "").strip()

        return response

    except Exception as e:
        return f"Error generating response: {str(e)}"


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Chatbot Assistant</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model status
        st.subheader("Model Status")
        if torch.cuda.is_available():
            st.success("✅ CUDA Available")
            st.info(f"GPU: {torch.cuda.get_device_name()}")
        else:
            st.warning("⚠️ Using CPU (slower)")

        # Generation parameters
        st.subheader("Generation Parameters")
        max_tokens = st.slider("Max New Tokens", 50, 300, 100)
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
        top_p = st.slider("Top P", 0.1, 1.0, 0.7)

        # Disclaimer
        st.subheader("⚠️ Medical Disclaimer")
        st.warning("""
        This chatbot is for informational purposes only. 
        Always consult with qualified healthcare professionals 
        for medical advice, diagnosis, or treatment.
        """)

    # Load model
    model, tokenizer, generation_config = load_model()

    if model is None:
        st.error("Failed to load the model. Please check your configuration.")
        return

    # Update generation config with sidebar values
    generation_config.max_new_tokens = max_tokens
    generation_config.temperature = temperature
    generation_config.top_p = top_p

    # Chat interface
    st.subheader("💬 Ask your medical question")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"><strong>Medical Bot:</strong> {message["content"]}</div>',
                        unsafe_allow_html=True)

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Enter your medical question:",
            placeholder="e.g., What is diabetes?",
            height=100
        )
        submit_button = st.form_submit_button("Send", use_container_width=True)

        if submit_button and user_input.strip():
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Generate response
            with st.spinner("Generating response..."):
                start_time = time.time()
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                response = generate_response(model, tokenizer, generation_config, user_input, device)
                end_time = time.time()

            # Add bot response to chat history
            st.session_state.messages.append({"role": "bot", "content": response})

            # Show generation time
            st.info(f"Response generated in {end_time - start_time:.2f} seconds")

            # Rerun to update the chat display
            st.rerun()

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Example questions
    st.subheader("💡 Example Questions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("What is diabetes?"):
            st.session_state.messages.append({"role": "user", "content": "What is diabetes?"})
            st.rerun()

    with col2:
        if st.button("What are symptoms of hypertension?"):
            st.session_state.messages.append({"role": "user", "content": "What are symptoms of hypertension?"})
            st.rerun()

    with col3:
        if st.button("How to prevent heart disease?"):
            st.session_state.messages.append({"role": "user", "content": "How to prevent heart disease?"})
            st.rerun()


if __name__ == "__main__":
    main()