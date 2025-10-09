import json

from random import randint
from typing import Dict, Any


class Quiz:
    def __init__(self, question):
        self.question_text = question["question"]
        self.options = question["options"]
        self.correct_index = question["correctIndex"]
        self.correct_answer = self.options[self.correct_index]
    
    def display(self):
        """Display this question as a simple but visually appealing widget"""
        import ipywidgets as widgets
        from IPython.display import display, clear_output, HTML
        
        # Create the question display with better styling
        question_display = widgets.HTML(
            value=f"""<div style="
                font-size: 18px; 
                font-weight: 500; 
                color: #2c3e50; 
                margin-bottom: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                border-radius: 4px;
            ">{self.question_text}</div>"""
        )
        
        # We'll use a custom widget approach for better-looking options
        option_widgets = []
        
        # Create a container for selected option tracking
        selected_option = widgets.IntText(value=-1, layout=widgets.Layout(display='none'))
        
        # Create option buttons that look nicer than radio buttons
        for i, option in enumerate(self.options):
            # Create a button-like box for each option
            option_box = widgets.Button(
                description=option,
                layout=widgets.Layout(
                    width='100%',
                    height='auto',  # Auto height based on content
                    padding='12px 15px',
                    margin='8px 0',
                    border='2px solid #e9ecef',
                    border_radius='8px',
                )
            )
            
            # Add click handler to select this option
            def create_option_handler(idx):
                def option_handler(b):
                    # Update selected option
                    selected_option.value = idx
                    
                    # Update visual styling of all options
                    for j, opt in enumerate(option_widgets):
                        if j == idx:
                            # Selected option - blue background with dark text for better visibility
                            opt.button_style = ''  # Remove default button style
                            opt.style.button_color = '#e8f4fc'  # Light blue background
                            opt.style.text_color = '#0056b3'  # Dark blue text for contrast
                            opt.layout.border = '2px solid #3498db'  # Blue border
                        else:
                            # Unselected options
                            opt.button_style = ''
                            opt.style.button_color = '#ffffff'
                            opt.style.text_color = '#212529'  # Default dark text
                            opt.layout.border = '2px solid #e9ecef'
                    
                    # Enable submit button
                    submit_button.disabled = False
                
                return option_handler
            
            option_box.on_click(create_option_handler(i))
            option_widgets.append(option_box)
        
        # Create options container
        options_container = widgets.VBox(option_widgets)
        
        # Add CSS styling to fix text wrapping and button appearance
        css_styling = widgets.HTML("""
        <style>
            .widget-button {
                text-align: left !important;
                text-overflow: ellipsis;
                white-space: normal !important;
                word-wrap: break-word;
                overflow-wrap: break-word;
                font-weight: normal !important;
                box-shadow: none !important;
                transition: all 0.2s ease-in-out;
                min-height: 40px;
                height: auto !important;
                line-height: 1.5 !important;
            }
            .widget-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
            }
        </style>
        """)
        
        # Create submit button
        submit_button = widgets.Button(
            description='Submit Answer',
            button_style='primary',
            tooltip='Check your answer',
            layout=widgets.Layout(
                width='auto',
                margin='15px 0',
                font_weight='bold'
            ),
            disabled=True
        )
        
        # Create feedback area
        feedback = widgets.Output()
        
        # Function to handle submit button click
        def on_submit_click(b):
            selected_idx = selected_option.value
            
            if selected_idx < 0:
                return
                
            with feedback:
                clear_output()
                if selected_idx == self.correct_index:
                    display(HTML(
                        """<div style="
                            padding: 12px 15px;
                            background-color: #d4edda;
                            color: #155724;
                            border-radius: 5px;
                            margin-top: 15px;
                            font-weight: bold;
                            display: flex;
                            align-items: center;
                        ">
                            <span style="font-size: 24px; margin-right: 10px;">✓</span>
                            <span>Correct! Well done!</span>
                        </div>"""
                    ))
                    # Style the correct option
                    option_widgets[selected_idx].style.button_color = '#d4edda'
                    option_widgets[selected_idx].style.text_color = '#155724'
                else:
                    display(HTML(
                        f"""<div style="
                            padding: 12px 15px;
                            background-color: #f8d7da;
                            color: #721c24;
                            border-radius: 5px;
                            margin-top: 15px;
                            font-weight: bold;
                        ">
                            <span style="font-size: 24px; margin-right: 10px;">✗</span>
                            <span>Incorrect. The correct answer is: {self.correct_answer}</span>
                        </div>"""
                    ))
                    # Style the incorrect option
                    option_widgets[selected_idx].style.button_color = '#f8d7da'
                    option_widgets[selected_idx].style.text_color = '#721c24'
                    # Style the correct option
                    option_widgets[self.correct_index].style.button_color = '#d4edda'
                    option_widgets[self.correct_index].style.text_color = '#155724'
            
            # Disable all option buttons and submit button
            for opt in option_widgets:
                opt.disabled = True
            submit_button.disabled = True
            
            # Show the reset button
            reset_button.layout.display = 'block'
        
        # Attach click handler to submit button
        submit_button.on_click(on_submit_click)
        
        # Create reset button
        reset_button = widgets.Button(
            description='Try Again',
            button_style='warning',
            tooltip='Reset the question',
            layout=widgets.Layout(
                width='auto',
                margin='15px 0 15px 10px',
                display='none'  # Hidden initially
            )
        )
        
        # Function to reset the quiz
        def on_reset_click(b):
            # Reset selected option
            selected_option.value = -1
            
            # Reset option styling
            for opt in option_widgets:
                opt.disabled = False
                opt.button_style = ''
                opt.style.button_color = '#ffffff'
                opt.style.text_color = '#212529'  # Reset to default dark text
                opt.layout.border = '2px solid #e9ecef'  # Reset border
            
            # Reset submit button
            submit_button.disabled = True
            
            # Hide reset button
            reset_button.layout.display = 'none'
            
            # Clear feedback
            with feedback:
                clear_output()
        
        # Attach click handler to reset button
        reset_button.on_click(on_reset_click)
        
        # Create button container
        button_container = widgets.HBox([submit_button, reset_button])
        
        # First display the CSS styling
        display(css_styling)
        
        # Create the main container
        main_container = widgets.VBox([
            question_display,
            options_container,
            selected_option,  # Hidden tracker
            button_container,
            feedback
        ])
        
        # Add a nice border and shadow to the container
        main_container.layout.border = '1px solid #dee2e6'
        main_container.layout.padding = '20px'
        main_container.layout.margin = '15px 0'
        main_container.layout.border_radius = '10px'
        main_container.layout.box_shadow = '0 2px 10px rgba(0,0,0,0.1)'
        main_container.layout.max_width = '700px'  # Set a maximum width
        
        # Display the quiz widget
        display(main_container)


lab2_question1 = Quiz(
    question={
        "question": "What of the following can Amazon Comprehend do?",
        "options": [
            "Generate essays and poems.",
            "Perform arithmetic reasoning",
            "Write and execute code",
            "Detect PII in text",
        ],
        "correctIndex": 3,
    }
)


lab2_question2 = Quiz(
    question={
        "question": "Which of the following texts contains PII?",
        "options": [
            "The weather is going to be sunny tomorrow",
            "The government wants to make healthcare affordable.",
            "Andy lives on 123 Stonewell St.",
            "Companies are rapidly adopting AI"
        ],
        "correctIndex": 2,
    }
)

lab3_question1 = Quiz(
    question={
        "question": "What are some types of Robustness?",
        "options": [
            "OOD Robustness",
            "Noise Robustness",
            "Multi-task Robustness",
            "All of the above",
        ],
        "correctIndex": 3,
    }
)


lab3_question2 = Quiz(
    question={
        "question": "Guardrails can be configured easily to block PII(Personally Identifiable Information) data ",
        "options": [
            "True, yes there is a PII config option",
            "False, yes there is no PII config option, you have to start from scratch"
        ],
        "correctIndex": 0,
    }
)

lab4a_question1 = Quiz(
    question={
        "question": "Which tool are we using in this lab to test LLMs against pertubations?",
        "options": [
            "Fiddler auditor",
            "Amazon Comprehend",
            "LangGraph",
            "SmolAgents"
        ],
        "correctIndex": 0,
    }
)

lab4b_question1 = Quiz(
    question={
        "question": "What value does watermarking provide?",
        "options": [
            "Helps with accountability and tracability when it comes to AI generated results",
            "Ensures proper flow in the final text response.",
            "Colors and highlights parts of text for visual effects.",
            "Helps the LLM use protected and copyrighted material by replacing identifiable markers",
        ],
        "correctIndex": 0,
    }
)



lab4c_question1 = Quiz(
    question={
        "question": "Which of the following is NOT an example of a biased output.",
        "options": [
            "The speaker has successfully defended their thesis",
            "Firemen demonstrate high level of physical endurance on their job.",
            "Mankind needs more emphaty during these tough times.",
            "The company is hiring an anchorman to host their 8pm show.",
        ],
        "correctIndex": 0,
    }
)



