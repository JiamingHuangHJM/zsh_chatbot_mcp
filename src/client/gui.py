import sys
import os
import random
from datetime import datetime
import tkinter as tk
from tkinter import scrolledtext, font
import threading
from client.client import ClaudeClient

DEBUG = True
def debug_print(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

try:
    from dotenv import load_dotenv
    load_dotenv()  # load environment variables from .env
    CLAUDE_AVAILABLE = True
except ImportError:
    print("Warning: Anthropic/MCP libraries not found. Please install required packages.")
    CLAUDE_AVAILABLE = False




class ZSHStyleChatbotGUI:
    def __init__(self, root):
        debug_print("Initializing GPT-5...")
        self.root = root
        self.root.title("GPT - 5")
        self.root.geometry("800x600")
        
        # Set user information
        self.user = os.getenv("USER", os.getenv("USERNAME", "user"))
        self.bot_name = "Claude"
        
        # Message history for conversation context
        self.message_history = []
        
        # Check if Claude is available
        if not CLAUDE_AVAILABLE:
            print("Error: Claude libraries not available. Please install required packages.")
            self.root.after(100, lambda: self.root.destroy())
            return
            
        # Initialize Claude client
        try:
            debug_print("Creating ClaudeClient instance")
            self.claude = ClaudeClient()
            debug_print("ClaudeClient created successfully")
        except Exception as e:
            print(f"Error initializing Claude: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.root.after(100, lambda: self.root.destroy())
            return
        
        # Set macOS-style appearance
        if sys.platform == "darwin":  # macOS
            try:
                from tkmacosx import Button
                self.use_tkmacosx = True
            except ImportError:
                self.use_tkmacosx = False
        else:
            self.use_tkmacosx = False
        
        # Create custom fonts
        try:
            self.terminal_font = font.Font(family="Menlo", size=12)
            self.prompt_font = font.Font(family="Menlo", size=12, weight="bold")
        except Exception as e:
            print(f"Error creating fonts: {str(e)}")
            # Use fallback fonts
            self.terminal_font = font.Font(family="TkFixedFont", size=12)
            self.prompt_font = font.Font(family="TkFixedFont", size=12, weight="bold")
        
        # Set user information
        self.user = os.getenv("USER", os.getenv("USERNAME", "user"))
        self.bot_name = "GPT-5"
        
        # Define colors (oh-my-zsh inspired)
        self.colors = {
            "bg": "#1E1E1E",  # Terminal background
            "fg": "#EFEFEF",  # Default text
            "prompt": "#8BE9FD",  # Prompt color (cyan)
            "user_input": "#F8F8F2",  # User text (white)
            "path": "#50FA7B",  # Path color (green)
            "time": "#BD93F9",  # Time color (purple)
            "bot_name": "#FF79C6",  # Bot name color (pink)
            "bot_response": "#F1FA8C"  # Bot responses (yellow)
        }
        
        # Configure the root window
        self.root.configure(bg=self.colors["bg"])
        
        # Create main frame
        self.main_frame = tk.Frame(root, bg=self.colors["bg"])
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create chat display area (with terminal look)
        self.chat_display = scrolledtext.ScrolledText(
            self.main_frame,
            bg=self.colors["bg"],
            fg=self.colors["fg"],
            insertbackground=self.colors["fg"],
            font=self.terminal_font,
            padx=10,
            pady=10,
            wrap=tk.WORD,
            height=20
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 10))
        self.chat_display.config(state=tk.DISABLED)
        
        # Create input frame (for prompt and entry)
        self.input_frame = tk.Frame(self.main_frame, bg=self.colors["bg"])
        self.input_frame.pack(fill=tk.X, expand=False, padx=0, pady=0)
        
        # Prompt label
        self.prompt_text = tk.StringVar()
        self.update_prompt()
        self.prompt_label = tk.Label(
            self.input_frame,
            textvariable=self.prompt_text,
            bg=self.colors["bg"],
            fg=self.colors["prompt"],
            font=self.prompt_font,
            justify=tk.LEFT,
            anchor="w"
        )
        self.prompt_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # Input entry
        self.user_input = tk.Entry(
            self.input_frame,
            bg=self.colors["bg"],
            fg=self.colors["user_input"],
            insertbackground=self.colors["user_input"],
            font=self.terminal_font,
            bd=0,
            highlightthickness=0
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=0, pady=0)
        self.user_input.bind("<Return>", self.process_input)
        self.user_input.bind("<Up>", self.show_history_up)
        self.user_input.bind("<Down>", self.show_history_down)
        self.user_input.bind("<Tab>", self.tab_complete)
        self.user_input.focus_set()
        
        # Command history
        self.history = []
        self.history_index = -1
        
        # Commands for tab completion
        self.commands = ["help", "exit", "quit", "clear", "time", "echo", "reset mcp"]
        
        # Initialize history and commands
        # (Bot name and user are already defined above)
        
        # Display welcome message
        self.display_bot_message(f"Welcome to {self.bot_name}! Type 'help' for available commands.")
        
        # Auto-update prompt every second
        self.update_prompt_periodically()
        
        debug_print("GUI initialization complete")
    
    def update_prompt(self):
        """Update the prompt with current time and path"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Create simpler prompt with just the time (no path or username)
        time_part = f"[{current_time}]"
        prompt = f"{time_part} ➜ "
        self.prompt_text.set(prompt)
        
    def update_prompt_periodically(self):
        """Update the prompt every second"""
        self.update_prompt()
        self.root.after(1000, self.update_prompt_periodically)
    
    def display_message(self, message, color=None):
        """Display a message in the chat display"""
        if color is None:
            color = self.colors["fg"]
            
        self.chat_display.config(state=tk.NORMAL)
        
        # Ensure we're at the beginning of a new line
        last_char = self.chat_display.get("end-2c", "end-1c") if self.chat_display.index("end-1c") != "1.0" else "\n"
        if last_char != '\n':
            self.chat_display.insert(tk.END, "\n")
            
        self.chat_display.insert(tk.END, message + "\n", f"color_{color}")
        self.chat_display.tag_configure(f"color_{color}", foreground=color)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def display_bot_message(self, message):
        """Display a message from the bot"""
        bot_prefix = f"[{self.bot_name}] "
        self.chat_display.config(state=tk.NORMAL)
        
        # Ensure we're at the beginning of a new line
        last_char = self.chat_display.get("end-2c", "end-1c") if self.chat_display.index("end-1c") != "1.0" else "\n"
        if last_char != '\n':
            self.chat_display.insert(tk.END, "\n")
        
        # Insert bot name with its color
        self.chat_display.insert(tk.END, bot_prefix, "bot_name")
        self.chat_display.tag_configure("bot_name", foreground=self.colors["bot_name"])
        
        # Insert message with its color
        self.chat_display.insert(tk.END, message + "\n", "bot_response")
        self.chat_display.tag_configure("bot_response", foreground=self.colors["bot_response"])
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def display_user_input(self, user_input):
        """Display the user's input in the chat display"""
        prompt = self.prompt_text.get()
        
        self.chat_display.config(state=tk.NORMAL)
        
        # Extract time part for coloring
        time_part = prompt.split("]")[0] + "]"
        
        # Insert full prompt with time
        self.chat_display.insert(tk.END, time_part + " ", "time_color")
        self.chat_display.tag_configure("time_color", foreground=self.colors["time"])
        
        # Insert the arrow
        self.chat_display.insert(tk.END, "➜ ", "prompt_color")
        self.chat_display.tag_configure("prompt_color", foreground=self.colors["prompt"])
        
        # Insert user input on the same line
        self.chat_display.insert(tk.END, user_input + "\n", "input_color")
        self.chat_display.tag_configure("input_color", foreground=self.colors["user_input"])
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    def start_progress_indicator(self, message="Thinking"):
        """Start a progress indicator animation"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Ensure we're at the beginning of a new line
        last_char = self.chat_display.get("end-2c", "end-1c") if self.chat_display.index("end-1c") != "1.0" else "\n"
        if last_char != '\n':
            self.chat_display.insert(tk.END, "\n")
        
        # Insert bot prefix and message
        bot_prefix = f"[{self.bot_name}] "
        self.chat_display.insert(tk.END, bot_prefix, "thinking_prefix")
        self.chat_display.tag_configure("thinking_prefix", foreground=self.colors["bot_name"])
        
        self.chat_display.insert(tk.END, message, "thinking_msg")
        self.chat_display.tag_configure("thinking_msg", foreground=self.colors["bot_name"])
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        # Initialize animation ID attribute if it doesn't exist
        if not hasattr(self, "animate_id"):
            self.animate_id = None
            
        # Set up the animation
        def animate_dots(count=0):
            if hasattr(self, "animate_id") and self.animate_id:
                self.root.after_cancel(self.animate_id)
                
            self.chat_display.config(state=tk.NORMAL)
            
            # Add dots based on count
            dots = "." * ((count % 8) + 1)
            
            # Find position right after the message
            start_index = self.chat_display.index("end-1c linestart")
            end_index = self.chat_display.index("end-1c lineend")
            
            # Find position after "Thinking"
            thinking_pos = self.chat_display.search(message, start_index, stopindex=end_index)
            if thinking_pos:
                after_thinking_pos = self.chat_display.index(f"{thinking_pos}+{len(message)}c")
                
                # Delete any previous dots
                self.chat_display.delete(after_thinking_pos, end_index)
                
                # Add new dots
                self.chat_display.insert(after_thinking_pos, dots, "thinking_dots")
                self.chat_display.tag_configure("thinking_dots", foreground=self.colors["bot_name"])
            
            self.chat_display.see(tk.END)
            self.chat_display.config(state=tk.DISABLED)
            
            # Schedule next animation frame
            self.animate_id = self.root.after(300, animate_dots, count+1)
            
        # Start the animation
        animate_dots()
    
    def stop_progress_indicator(self):
        """Stop the progress indicator animation (but don't remove the text)"""
        # Cancel any pending animation
        if hasattr(self, "animate_id") and self.animate_id:
            self.root.after_cancel(self.animate_id)
            self.animate_id = None
    
    def process_input(self, event=None):
        """Process user input"""
        user_input = self.user_input.get().strip()
        if not user_input:
            return
        
        # Add to history
        self.history.append(user_input)
        self.history_index = -1
        
        # Display in chat
        self.display_user_input(user_input)
        
        # Clear input field
        self.user_input.delete(0, tk.END)
        
        # Process commands
        if user_input.lower() in ["exit", "quit", "bye", "q"]:
            self.handle_exit()
        elif user_input.lower() in ["help", "?", "-h", "--help"]:
            self.handle_help()
        elif user_input.lower() == "clear":
            self.handle_clear()
        elif user_input.lower() == "time":
            self.handle_time()
        elif user_input.lower().startswith("echo "):
            self.handle_echo(user_input)
        elif user_input.lower().startswith("mcp "):
            # Special command to connect to MCP server
            server_path = user_input[4:].strip()
            self.connect_to_mcp(server_path)
        elif user_input.lower() == "reset mcp":
            # Add a command to force MCP reset
            self.handle_reset_mcp()
        else:
            # Process with Claude
            self.process_claude_query(user_input)
    
    def handle_reset_mcp(self):
        """Handle the reset mcp command"""
        self.display_bot_message("Performing full MCP reset and reconnection...")
        
        # Start a progress indicator
        self.start_progress_indicator("Resetting MCP connection")
        
        def after_reset():
            # Stop the animation
            self.stop_progress_indicator()
            self.display_bot_message("MCP connection has been reset. Try your query again.")
            
        # Perform the reset in a thread to avoid blocking UI
        def reset_thread():
            try:
                if hasattr(self, 'claude'):
                    self.claude.full_reset_and_reconnect()
                # Use after to update UI from main thread
                self.root.after(0, after_reset)
            except Exception as e:
                debug_print(f"Error in reset: {str(e)}")
                # Use after to update UI from main thread
                self.root.after(0, lambda: self.display_bot_message(f"Error resetting MCP: {str(e)}"))
                
        # Start the reset in a thread
        threading.Thread(target=reset_thread).start()
    
    def handle_exit(self):
        """Handle exit command"""
        farewell_messages = ["Goodbye!", "See you later!", "Bye!", "Until next time!"]
        self.display_bot_message(random.choice(farewell_messages))
        self.root.after(1000, self.root.destroy)
    
    def handle_help(self):
        """Handle help command"""
        help_text = """Available Commands:
- help, ?, -h, --help : Show this help message
- exit, quit, bye, q  : Exit the chatbot
- clear               : Clear the screen
- time                : Display current date and time
- echo <message>      : Echo back a message
- mcp <path>          : Connect to Claude via MCP server script
- reset mcp           : Force reset of MCP connection

For any other input, the chatbot will try to generate a response."""
        
        if hasattr(self, 'claude') and hasattr(self.claude, 'mcp_enabled') and self.claude.mcp_enabled:
            claude_status = "Claude is CONNECTED and will respond to your messages."
        else:
            claude_status = "Claude is CONNECTED without MCP. Use 'mcp <path>' to enable MCP features."
            
        help_text += f"\n\n{claude_status}"
        self.display_bot_message(help_text)
    
    def handle_clear(self):
        """Handle clear command"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.display_bot_message("Display cleared.")
    
    def handle_time(self):
        """Handle time command"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.display_bot_message(f"Current time: {current_time}")
    
    def handle_echo(self, command):
        """Handle echo command"""
        message = command[5:]  # Remove 'echo ' prefix
        self.display_bot_message(message)
    
    def connect_to_mcp(self, server_path):
        """Connect to MCP server"""
        
        # Start a progress indicator
        self.start_progress_indicator("Connecting to MCP server")
        
        # Define a callback to handle MCP connection result
        def mcp_connection_callback(success, message):
            # Stop the animation but keep the text
            self.root.after(0, self.stop_progress_indicator)
            
            if success:
                # Display success message
                self.root.after(50, lambda msg=message: self.display_bot_message(msg))
                self.root.after(100, lambda: self.display_bot_message("MCP is now enabled! Tools are available for use."))
            else:
                # Display error message
                self.root.after(50, lambda msg=message: self.display_bot_message(f"Error connecting to MCP server: {msg}"))
                self.root.after(100, lambda: self.display_bot_message("Using Claude without MCP tools instead."))
        
        # Connect to MCP
        def connect_thread():
            try:
                # Make sure MCP client is initialized first
                self.claude.init_mcp_if_needed()
                
                # First, do a full reset if needed
                if hasattr(self, 'claude') and hasattr(self.claude, 'mcp_enabled') and self.claude.mcp_enabled:
                    debug_print("Resetting existing MCP connection before connecting to new server")
                    self.claude._cleanup_mcp_resources()
                    self.claude._start_mcp_loop()
                    
                # Connect to the new server
                self.claude.connect_to_mcp(server_path, mcp_connection_callback)
            except Exception as e:
                import traceback
                debug_print(f"Critical error in connect_thread: {str(e)}")
                debug_print(traceback.format_exc())
                self.root.after(0, lambda: mcp_connection_callback(False, f"Critical error: {str(e)}"))
        
        # Start connection in a new thread
        threading.Thread(target=connect_thread).start()
    
    def process_claude_query(self, query):
        """Process a query with Claude"""
        # Start a progress indicator
        self.start_progress_indicator("Thinking")
        
        def query_thread():
            try:
                # Make sure everything is initialized
                if hasattr(self, 'claude') and hasattr(self.claude, 'mcp_enabled') and self.claude.mcp_enabled:
                    self.claude.init_mcp_if_needed()
                
                # Print debug info
                debug_print(f"Processing query: {query}")
                
                # Get response (this process now manages its own async context)
                response = self.claude.process_query(query, self.message_history)
                
                debug_print(f"Got response: {response[:50]}...")
                
                # Use after() to schedule UI updates on the main thread
                # First stop the animation (but keep the text)
                self.root.after(0, self.stop_progress_indicator)
                
                # Then display the response on a new line
                self.root.after(50, lambda r=response: self.display_bot_message(r))
                
                # Update message history with this exchange
                if self.message_history and self.message_history[-1]["role"] == "user":
                    # Replace the last user message if it exists
                    self.message_history[-1] = {"role": "user", "content": query}
                else:
                    # Add the user message
                    self.message_history.append({"role": "user", "content": query})
                    
                # Add the assistant's response to history
                self.message_history.append({"role": "assistant", "content": response})
                
            except Exception as e:
                # Print the full exception for debugging
                import traceback
                debug_print(f"Error processing query: {str(e)}")
                debug_print(traceback.format_exc())
                
                # Stop the animation but keep the text
                self.root.after(0, self.stop_progress_indicator)
                
                # Display the error on a new line
                error_msg = f"Error: {str(e)}"
                self.root.after(50, lambda msg=error_msg: self.display_bot_message(msg))
        
        # Run in a separate thread to avoid blocking the UI
        threading.Thread(target=query_thread).start()
    
    def show_history_up(self, event=None):
        """Navigate up through command history"""
        if not self.history:
            return "break"
        
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.user_input.delete(0, tk.END)
            self.user_input.insert(0, self.history[-(self.history_index+1)])
        
        return "break"  # Prevent default handling
    
    def show_history_down(self, event=None):
        """Navigate down through command history"""
        if self.history_index > 0:
            self.history_index -= 1
            self.user_input.delete(0, tk.END)
            self.user_input.insert(0, self.history[-(self.history_index+1)])
        elif self.history_index == 0:
            self.history_index = -1
            self.user_input.delete(0, tk.END)
        
        return "break"  # Prevent default handling
    
    def tab_complete(self, event=None):
        """Provide tab completion for commands"""
        current_text = self.user_input.get()
        matches = [cmd for cmd in self.commands if cmd.startswith(current_text)]
        
        if len(matches) == 1:
            # Exact match, complete
            self.user_input.delete(0, tk.END)
            self.user_input.insert(0, matches[0])
        elif len(matches) > 1:
            # Show possibilities
            self.display_message("Possible commands: " + ", ".join(matches), self.colors["fg"])
        
        return "break"  # Prevent default handling

def start_gui():
    # Add debug output at main startup
    print("Starting ZSH-Style Claude Terminal")
    
    try:
        root = tk.Tk()
        print("Tkinter initialized")
        
        # Create the application
        try:
            print("Creating application instance")
            app = ZSHStyleChatbotGUI(root)
            print("Application instance created")
        except Exception as e:
            print(f"Failed to create application instance: {str(e)}")
            import traceback
            print(traceback.format_exc())
            root.destroy()
            return
        
        # Handle command line arguments for MCP server
        if len(sys.argv) > 1 and hasattr(app, 'claude'):
            # If a server path is provided as argument, connect to MCP after a short delay
            server_path = sys.argv[1]
            print(f"Initializing with MCP server path: {server_path}")
            
            # Schedule MCP connection after a delay to allow GUI to initialize fully
            def delayed_connect():
                try:
                    # Initialize MCP loop first
                    app.claude.init_mcp_if_needed()
                    # Then connect
                    app.connect_to_mcp(server_path)
                except Exception as e:
                    print(f"Error in delayed MCP connection: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
            
            # Use a longer delay to ensure the GUI is fully initialized
            app.root.after(2000, delayed_connect)
        
        # Start with a welcome message
        if hasattr(app, 'claude'):
            app.display_bot_message("Welcome to the GPT-5 chatbot! Type 'help' for available commands.")
            
        # Clean up when the app closes
        def on_closing():
            print("Application closing")
            if hasattr(app, 'claude'):
                # Properly clean up MCP client
                app.claude.cleanup()
                    
            # Always destroy the root window
            root.destroy()
            
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start the main loop
        print("Starting Tkinter main loop")
        root.mainloop()
        print("Tkinter main loop ended")
        
    except Exception as e:
        print(f"Critical error starting application: {str(e)}")
        import traceback
        print(traceback.format_exc())
