import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output, Markdown
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import warnings

warnings.filterwarnings("ignore")

def create_memory_demo_ui(llm):
    # Apply styling for a cleaner UI
    display(HTML("""
    <style>
        .memory-demo-container { max-width: 1000px; margin: 0 auto; }
        .section-header { color: #2c3e50; font-weight: 600; margin-top: 20px; 
                         margin-bottom: 12px; padding-bottom: 8px; border-bottom: 2px solid #ebf5fb; }
        .widget-button button { background-color: #3498db !important; color: white !important; 
                               border: none !important; border-radius: 4px !important; }
        .widget-button button:hover:enabled { background-color: #2980b9 !important; }
        .widget-button button:disabled { background-color: #bdc3c7 !important; opacity: 0.7 !important; }
        .conversation-area { background-color: #f8f9fa; border-radius: 6px; padding: 15px; 
                            border: 1px solid #e0e0e0; font-family: sans-serif; }
        .memory-area { background-color: #f0f7fb; border-radius: 6px; padding: 15px; 
                      border: 1px solid #d1e3f0; font-family: monospace; }
        .user-message { background-color: #e8f4fd; border-radius: 18px; padding: 10px 15px; 
                       margin: 5px 0; display: inline-block; max-width: 80%; }
        .ai-message { background-color: #f0f0f0; border-radius: 18px; padding: 10px 15px; 
                     margin: 5px 0 15px auto; display: inline-block; max-width: 80%; }
        .message-container { margin: 10px 0; }
        .user-container { text-align: left; }
        .ai-container { text-align: right; }
        .memory-content { white-space: pre-wrap; }
        .memory-header { color: #3498db; font-weight: bold; margin-top: 10px; }
        .memory-item { margin: 5px 0; padding: 5px; border-bottom: 1px solid #e0e0e0; }
    </style>
    """))
    
    # Create memory store and session history
    session_id = "demo_session"
    memory_store = {}
    
    def get_session_history(session_id):
        if session_id not in memory_store:
            memory_store[session_id] = InMemoryChatMessageHistory()
        return memory_store[session_id]
    
    # Create a prompt template that includes chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Create the chain with history
    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: get_session_history(session_id),
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # Create widgets
    prompt_input = widgets.Text(
        value='',
        placeholder='Type your prompt here and press Enter or click Submit...',
        description='',
        layout=widgets.Layout(width='100%')
    )
    
    submit_button = widgets.Button(
        description='Submit',
        icon='paper-plane',
        button_style='primary',
        layout=widgets.Layout(width='150px')
    )
    
    conversation_output = widgets.Output(layout=widgets.Layout(
        width='100%',
        min_height='300px',
        max_height='500px',
        overflow='auto'
    ))
    
    memory_output = widgets.Output(layout=widgets.Layout(
        width='100%',
        min_height='300px',
        max_height='500px',
        overflow='auto'
    ))
    
    # History tracking
    conversation_history = []
    
    # Function to format the conversation nicely
    def format_conversation():
        with conversation_output:
            clear_output(wait=True)
            for user_msg, ai_msg in conversation_history:
                display(HTML(f"""
                <div class="message-container user-container">
                    <div class="user-message">👤 <strong>User:</strong> {user_msg}</div>
                </div>
                <div class="message-container ai-container">
                    <div class="ai-message">🤖 <strong>AI:</strong> {ai_msg.content}</div>
                </div>
                """))
    
    # Function to update the memory display
    def update_memory_display():
        with memory_output:
            clear_output(wait=True)
            
            # Get the current history
            history = get_session_history(session_id).messages
            
            # Format the history as a string
            buffer_content = ""
            for msg in history:
                if isinstance(msg, HumanMessage):
                    buffer_content += f"Human: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    buffer_content += f"AI: {msg.content}\n"
            
            display(HTML(f"""
            <div class="memory-content memory-item">{buffer_content}</div>
            """))
    
    # Handle submission
    def on_submit(b=None):
        prompt = prompt_input.value
        if not prompt.strip():
            return
            
        # Clear the input field
        prompt_input.value = ''
        
        # Get response from the chain
        response = chain_with_history.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}}
        )
        
        # Add to history
        conversation_history.append((prompt, response))
        
        # Update displays
        format_conversation()
        update_memory_display()
    
    # Connect the button click and Enter key to the handler
    submit_button.on_click(on_submit)
    prompt_input.on_submit(on_submit)
    
    # Create layout
    input_box = widgets.HBox([
        prompt_input, 
        widgets.Box([submit_button], layout=widgets.Layout(display='flex', align_items='center'))
    ], layout=widgets.Layout(width='100%', margin='15px 0'))
    
    # Initialize displays
    format_conversation()
    update_memory_display()
    
    # Assemble the UI
    return widgets.VBox([
        widgets.HTML("<h2 class='section-header'>Memory Modules Widget</h2>"),
        widgets.HTML("""
        <p style="color:#555; margin-bottom:20px;">
            This demo shows how memory modules work. Type prompts to interact with the AI 
            and observe how the memory buffer keeps track of the conversation history.
        </p>
        """),
        widgets.HBox([
            widgets.VBox([
                widgets.HTML("<h3 class='section-header'>Conversation</h3>"),
                widgets.Box([conversation_output], layout=widgets.Layout(
                    border='1px solid #e0e0e0', 
                    border_radius='6px',
                    padding='0',
                    width='100%',
                    class_='conversation-area'
                ))
            ], layout=widgets.Layout(width='50%', padding='0 10px 0 0')),
            
            widgets.VBox([
                widgets.HTML("<h3 class='section-header'>🧠 MEMORY BUFFER CONTENT</h3>"),
                widgets.Box([memory_output], layout=widgets.Layout(
                    border='1px solid #d1e3f0', 
                    border_radius='6px',
                    padding='0',
                    width='100%',
                    class_='memory-area'
                ))
            ], layout=widgets.Layout(width='50%', padding='0 0 0 10px'))
        ], layout=widgets.Layout(width='100%')),
        
        widgets.HTML("<h3 class='section-header'>New Prompt</h3>"),
        input_box
    ], layout=widgets.Layout(
        padding='25px', 
        border='1px solid #e0e0e0', 
        border_radius='8px', 
        width='100%', 
        max_width='1000px', 
        margin='0 auto',
        class_='memory-demo-container'
    ))


def create_configurable_memory_demo_ui(llm):
    # Apply styling for a cleaner UI
    display(HTML("""
    <style>
        .memory-demo-container { max-width: 1000px; margin: 0 auto; }
        .section-header { color: #2c3e50; font-weight: 600; margin-top: 20px; 
                         margin-bottom: 12px; padding-bottom: 8px; border-bottom: 2px solid #ebf5fb; }
        .widget-button button { background-color: #3498db !important; color: white !important; 
                               border: none !important; border-radius: 4px !important; }
        .widget-button button:hover:enabled { background-color: #2980b9 !important; }
        .widget-button button:disabled { background-color: #bdc3c7 !important; opacity: 0.7 !important; }
        .conversation-area { background-color: #f8f9fa; border-radius: 6px; padding: 15px; 
                            border: 1px solid #e0e0e0; font-family: sans-serif; }
        .memory-area { background-color: #f0f7fb; border-radius: 6px; padding: 15px; 
                      border: 1px solid #d1e3f0; font-family: monospace; }
        .user-message { background-color: #e8f4fd; border-radius: 18px; padding: 10px 15px; 
                       margin: 5px 0; display: inline-block; max-width: 80%; }
        .ai-message { background-color: #f0f0f0; border-radius: 18px; padding: 10px 15px; 
                     margin: 5px 0 15px auto; display: inline-block; max-width: 80%; }
        .message-container { margin: 10px 0; }
        .user-container { text-align: left; }
        .ai-container { text-align: right; }
        .memory-content { white-space: pre-wrap; }
        .memory-header { color: #3498db; font-weight: bold; margin-top: 10px; }
        .memory-item { margin: 5px 0; padding: 5px; border-bottom: 1px solid #e0e0e0; }
        .parameter-container { margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 6px; }
        .parameter-label { font-weight: bold; margin-bottom: 5px; }
    </style>
    """))
    
    # Memory module selection
    memory_type_dropdown = widgets.Dropdown(
        options=[
            'ConversationBufferMemory',
            'ConversationBufferWindowMemory',
            'ConversationSummaryMemory'
        ],
        value='ConversationBufferMemory',
        description='Memory Type:',
        layout=widgets.Layout(width='350px')
    )
    
    # Parameter widgets for different memory types
    k_param = widgets.IntText(
        value=5,
        description='k (window size):',
        disabled=True,
        layout=widgets.Layout(width='200px')
    )
    
    # Apply button to update memory configuration
    apply_button = widgets.Button(
        description='Apply Memory Settings',
        button_style='primary',
        icon='check',
        layout=widgets.Layout(width='200px')
    )
    
    # Container for parameter widgets
    param_container = widgets.VBox([
        widgets.HTML("<p class='parameter-label'>Memory Parameters:</p>"),
        k_param
    ], layout=widgets.Layout(
        margin='10px 0',
        padding='10px',
        border='1px solid #e0e0e0',
        border_radius='6px',
        class_='parameter-container'
    ))
    
    # Session ID for memory
    session_id = "configurable_demo_session"
    memory_store = {}
    
    # Create a prompt template that includes chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Function to get session history based on memory type
    def get_session_history(session_id):
        if session_id not in memory_store:
            memory_store[session_id] = InMemoryChatMessageHistory()
        return memory_store[session_id]
    
    # Create the chain with history
    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: get_session_history(session_id),
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # Create widgets for conversation
    prompt_input = widgets.Text(
        value='',
        placeholder='Type your prompt here and press Enter or click Submit...',
        description='',
        layout=widgets.Layout(width='100%')
    )
    
    submit_button = widgets.Button(
        description='Submit',
        icon='paper-plane',
        button_style='primary',
        layout=widgets.Layout(width='150px')
    )
    
    reset_button = widgets.Button(
        description='Reset Conversation',
        icon='refresh',
        button_style='warning',
        layout=widgets.Layout(width='180px')
    )
    
    conversation_output = widgets.Output(layout=widgets.Layout(
        width='100%',
        min_height='300px',
        max_height='500px',
        overflow='auto'
    ))
    
    memory_output = widgets.Output(layout=widgets.Layout(
        width='100%',
        min_height='300px',
        max_height='500px',
        overflow='auto'
    ))
    
    # History tracking
    conversation_history = []
    
    # Function to update visible parameters based on memory type
    def update_parameter_visibility(memory_type):
        if memory_type == 'ConversationBufferMemory':
            k_param.disabled = True
        elif memory_type == 'ConversationBufferWindowMemory':
            k_param.disabled = False
        elif memory_type == 'ConversationSummaryMemory':
            k_param.disabled = True
    
    # Update parameters when memory type changes
    def on_memory_type_change(change):
        update_parameter_visibility(change.new)
    
    memory_type_dropdown.observe(on_memory_type_change, names='value')
    
    # Function to format the conversation nicely
    def format_conversation():
        with conversation_output:
            clear_output(wait=True)
            for user_msg, ai_msg in conversation_history:
                display(HTML(f"""
                <div class="message-container user-container">
                    <div class="user-message">👤 <strong>User:</strong> {user_msg}</div>
                </div>
                <div class="message-container ai-container">
                    <div class="ai-message">🤖 <strong>AI:</strong> {ai_msg.content}</div>
                </div>
                """))
    
    # Function to update the memory display
    def update_memory_display():
        with memory_output:
            clear_output(wait=True)
            
            memory_type = memory_type_dropdown.value
            history = get_session_history(session_id).messages
            
            # Format the history based on memory type
            if memory_type == 'ConversationBufferMemory':
                # Show all messages
                buffer_content = ""
                for msg in history:
                    if isinstance(msg, HumanMessage):
                        buffer_content += f"Human: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        buffer_content += f"AI: {msg.content}\n"
                
                display(HTML(f"""
                <div class="memory-header">Buffer Content:</div>
                <div class="memory-content memory-item">{buffer_content}</div>
                """))
                
            elif memory_type == 'ConversationBufferWindowMemory':
                # Show only the last k exchanges
                k = k_param.value
                window_size = min(k * 2, len(history))  # k exchanges = 2k messages
                recent_messages = history[-window_size:]
                
                buffer_content = ""
                for msg in recent_messages:
                    if isinstance(msg, HumanMessage):
                        buffer_content += f"Human: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        buffer_content += f"AI: {msg.content}\n"
                
                display(HTML(f"""
                <div class="memory-header">Window Buffer (k={k_param.value}):</div>
                <div class="memory-content memory-item">{buffer_content}</div>
                """))
                
            elif memory_type == 'ConversationSummaryMemory':
                # For summary memory, we'll create a summary of the conversation
                # This is a simplified version - in a real implementation, we'd use the LLM to generate a summary
                
                # Count the exchanges
                num_exchanges = len(history) // 2
                
                # Create a simple summary
                summary = f"This conversation has {num_exchanges} exchanges between the human and AI."
                if num_exchanges > 0:
                    summary += f"\nIt started with the human saying: \"{history[0].content if history else ''}\"" 
                    if len(history) >= 2:
                        summary += f"\nThe most recent exchange was about: \"{history[-2].content if len(history) >= 2 else ''}\""
                
                display(HTML(f"""
                <div class="memory-header">Summary:</div>
                <div class="memory-content memory-item">{summary}</div>
                """))
    
    # Handle memory configuration update
    def on_apply_settings(b):
        nonlocal memory_store, session_id
        
        # Reset the memory store for the session
        memory_store[session_id] = InMemoryChatMessageHistory()
        
        # Reset conversation history
        conversation_history.clear()
        
        # Update displays
        format_conversation()
        update_memory_display()
        
        # Show confirmation
        with memory_output:
            clear_output(wait=True)
            display(HTML(f"""
            <div style="color: green; font-weight: bold; padding: 10px; background-color: #e8f5e9; border-radius: 4px;">
                Memory settings applied successfully! Memory type: {memory_type_dropdown.value}
            </div>
            """))
    
    # Handle submission
    def on_submit(b=None):
        prompt = prompt_input.value
        if not prompt.strip():
            return
            
        # Clear the input field
        prompt_input.value = ''
        
        try:
            # Get response from the chain
            response = chain_with_history.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": session_id}}
            )
            
            # Add to history
            conversation_history.append((prompt, response))
            
            # Update displays
            format_conversation()
            update_memory_display()
        except Exception as e:
            with conversation_output:
                clear_output(wait=True)
                display(HTML(f"""
                <div style="color: red; font-weight: bold; padding: 10px; background-color: #ffebee; border-radius: 4px;">
                    Error: {str(e)}
                </div>
                """))
    
    # Handle reset
    def on_reset(b):
        nonlocal memory_store, session_id, conversation_history
        
        # Reset conversation history
        conversation_history = []
        
        # Reset the memory store for the session
        memory_store[session_id] = InMemoryChatMessageHistory()
        
        # Update displays
        format_conversation()
        update_memory_display()
    
    # Connect the button clicks and Enter key to the handlers
    submit_button.on_click(on_submit)
    prompt_input.on_submit(on_submit)
    apply_button.on_click(on_apply_settings)
    reset_button.on_click(on_reset)
    
    # Create layout
    memory_config_section = widgets.VBox([
        widgets.HTML("<h3 class='section-header'>Memory Configuration</h3>"),
        widgets.HBox([memory_type_dropdown, apply_button], layout=widgets.Layout(align_items='center')),
        param_container
    ])
    
    input_box = widgets.HBox([
        prompt_input, 
        widgets.Box([submit_button], layout=widgets.Layout(display='flex', align_items='center'))
    ], layout=widgets.Layout(width='100%', margin='15px 0'))
    
    # Initialize displays
    update_parameter_visibility(memory_type_dropdown.value)
    format_conversation()
    update_memory_display()
    
    # Assemble the UI
    return widgets.VBox([
        widgets.HTML("<h2 class='section-header'>Configurable Memory Modules Widget</h2>"),
        widgets.HTML("""
        <p style="color:#555; margin-bottom:20px;">
            This demo allows you to select and configure different memory modules. Choose a memory type,
            set its parameters, and click "Apply Memory Settings" to update the configuration.
        </p>
        """),
        memory_config_section,
        widgets.HBox([
            widgets.VBox([
                widgets.HTML("<h3 class='section-header'>Conversation</h3>"),
                widgets.Box([conversation_output], layout=widgets.Layout(
                    border='1px solid #e0e0e0', 
                    border_radius='6px',
                    padding='0',
                    width='100%',
                    class_='conversation-area'
                )),
                widgets.HBox([reset_button], layout=widgets.Layout(justify_content='flex-end', margin='10px 0'))
            ], layout=widgets.Layout(width='50%', padding='0 10px 0 0')),
            
            widgets.VBox([
                widgets.HTML("<h3 class='section-header'>🧠 MEMORY CONTENT</h3>"),
                widgets.Box([memory_output], layout=widgets.Layout(
                    border='1px solid #d1e3f0', 
                    border_radius='6px',
                    padding='0',
                    width='100%',
                    class_='memory-area'
                ))
            ], layout=widgets.Layout(width='50%', padding='0 0 0 10px'))
        ], layout=widgets.Layout(width='100%')),
        
        widgets.HTML("<h3 class='section-header'>New Prompt</h3>"),
        input_box
    ], layout=widgets.Layout(
        padding='25px', 
        border='1px solid #e0e0e0', 
        border_radius='8px', 
        width='100%', 
        max_width='1000px', 
        margin='0 auto',
        class_='memory-demo-container'
    ))