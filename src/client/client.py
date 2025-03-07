import time
import asyncio
import threading

DEBUG = True
def debug_print(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

try:
    from anthropic import Anthropic
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from dotenv import load_dotenv
    load_dotenv()  
    CLAUDE_AVAILABLE = True
except ImportError:
    print("Warning: Anthropic/MCP libraries not found. Please install required packages.")
    CLAUDE_AVAILABLE = False



class ClaudeClient:
    """Client to interact with Claude with or without MCP"""
    def __init__(self):
        debug_print("Initializing ClaudeClient")
        if not CLAUDE_AVAILABLE:
            raise ImportError("Claude libraries not installed. Please install the required packages.")
            
        self.anthropic = Anthropic()
        self.mcp_enabled = False
        self.model = "claude-3-5-sonnet-20241022" 
        
        # MCP connection variables
        self.mcp_loop = None
        self.mcp_thread = None
        self.mcp_running = False
        self.mcp_session = None
        self.exit_stack = None
        self.mcp_lock = threading.Lock()  # Add a lock for thread safety
        self.server_script_path = None  # Store the server path for reconnection
        
        # Initialize without starting MCP loop automatically
        # We'll start it only when needed
        debug_print("ClaudeClient initialized without starting MCP loop")
    
    def init_mcp_if_needed(self):
        """Initialize MCP only if needed"""
        if not self.mcp_running and not self.mcp_thread:
            debug_print("Starting MCP loop on-demand")
            self._start_mcp_loop()
    
    def _start_mcp_loop(self):
        """Start a dedicated event loop for MCP operations"""
        debug_print("Attempting to start MCP loop")
        # First check if we already have a running loop
        already_running = False
        
        with self.mcp_lock:  
            if self.mcp_thread is not None and self.mcp_thread.is_alive() and self.mcp_running:
                debug_print("MCP thread already running, not starting a new one")
                already_running = True
        
        if already_running:
            return
            
        # Clean up any existing resources first
        self._cleanup_mcp_resources()
            
        def run_event_loop():
            debug_print("Starting MCP event loop thread")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            with self.mcp_lock:
                self.mcp_loop = loop
                self.mcp_running = True
            
            debug_print("MCP loop now running")
            try:
                loop.run_forever()
                debug_print("MCP event loop exited normally")
            except Exception as e:
                debug_print(f"MCP event loop exited with error: {str(e)}")
            finally:
                with self.mcp_lock:
                    self.mcp_running = False
                    if not loop.is_closed():
                        loop.close()
                    debug_print("MCP event loop closed")
                
        # Create thread but don't start it in __init__
        new_thread = threading.Thread(target=run_event_loop, daemon=True)
        
        with self.mcp_lock:
            self.mcp_thread = new_thread
            
        # Start the thread outside the lock
        self.mcp_thread.start()
        
        # Give the thread a moment to start
        debug_print("Waiting for MCP thread to initialize")
        time.sleep(0.3)  # Shorter wait
    
    def _cleanup_mcp_resources(self):
        """Clean up MCP resources before creating new ones"""
        debug_print("Cleaning up MCP resources")
        with self.mcp_lock:
            # Close MCP session and exit stack first
            if self.exit_stack or self.mcp_session:
                debug_print("Found existing MCP session to clean up")
                # We need to schedule this in the current loop if it's running
                # Otherwise we'll do it in a new temporary loop
                if self.mcp_loop and not self.mcp_loop.is_closed() and self.mcp_running:
                    try:
                        debug_print("Scheduling session cleanup in existing loop")
                        async def cleanup_session():
                            try:
                                if self.exit_stack:
                                    debug_print("Closing MCP session in cleanup")
                                    await self.exit_stack.aclose()
                            except Exception as e:
                                debug_print(f"Error closing MCP session: {str(e)}")
                            finally:
                                self.exit_stack = None
                                self.mcp_session = None
                                
                        # Run cleanup in the loop
                        future = asyncio.run_coroutine_threadsafe(cleanup_session(), self.mcp_loop)
                        try:
                            # Wait with timeout
                            future.result(timeout=1.0)  # Shorter timeout
                        except Exception as e:
                            debug_print(f"Error waiting for session cleanup: {str(e)}")
                    except Exception as e:
                        debug_print(f"Failed to schedule session cleanup: {str(e)}")
                
                # Reset these variables regardless of cleanup success
                self.exit_stack = None
                self.mcp_session = None
                
            # Close any existing event loop
            if self.mcp_loop and not self.mcp_loop.is_closed():
                try:
                    debug_print("Stopping existing MCP loop")
                    self.mcp_loop.call_soon_threadsafe(self.mcp_loop.stop)
                except RuntimeError:
                    # Loop already closed, that's okay
                    debug_print("Failed to stop MCP loop - already closed")
                    pass
                    
            # Join thread if it exists
            if self.mcp_thread and self.mcp_thread.is_alive():
                try:
                    debug_print("Joining MCP thread")
                    self.mcp_thread.join(timeout=1.0)  # Shorter timeout
                except Exception as e:
                    debug_print(f"Error joining MCP thread: {str(e)}")
                
            # Reset all variables
            self.mcp_loop = None
            self.mcp_thread = None
            self.mcp_running = False
            self.mcp_enabled = False
            debug_print("MCP resources cleanup complete")
    
    def full_reset_and_reconnect(self):
        """Complete reset of MCP connection and attempt to reconnect"""
        debug_print("Performing full MCP reset and reconnection")
        
        # Store the current server path
        current_path = self.server_script_path
        
        # Complete cleanup
        self._cleanup_mcp_resources()
        
        # Start a new loop
        self._start_mcp_loop()
        
        # If we have a server path, try to reconnect
        if current_path:
            debug_print(f"Attempting to reconnect to {current_path}")
            # Create a simple callback for logging
            def reconnect_callback(success, message):
                if success:
                    debug_print(f"Reconnection successful: {message}")
                else:
                    debug_print(f"Reconnection failed: {message}")
            
            # Connect to the server
            self.connect_to_mcp(current_path, reconnect_callback)
            
            # Give it a moment to connect
            time.sleep(0.5)  # Shorter wait
    
    def _check_and_restart_mcp_loop(self):
        """Check if MCP loop is running and restart if needed"""
        with self.mcp_lock:
            if not self.mcp_running or not self.mcp_thread or not self.mcp_thread.is_alive() or \
               not self.mcp_loop or self.mcp_loop.is_closed():
                debug_print("MCP loop not running or closed, restarting...")
                loop_running = False
            else:
                loop_running = True
                
        if not loop_running:
            # Start outside the lock
            self._start_mcp_loop()
            return True
        
        return False
    
    def _stop_mcp_loop(self):
        """Stop the MCP event loop"""
        debug_print("Stopping MCP event loop")
        with self.mcp_lock:
            loop_running = (self.mcp_loop and not self.mcp_loop.is_closed() and 
                          self.mcp_running and self.mcp_thread and 
                          self.mcp_thread.is_alive())
        
        if loop_running:
            try:
                debug_print("Attempting to stop MCP loop via call_soon_threadsafe")
                self.mcp_loop.call_soon_threadsafe(self.mcp_loop.stop)
            except RuntimeError:
                # Loop already closed, that's okay
                debug_print("Failed to stop MCP loop - probably already closed")
                pass
                
            try:
                debug_print("Waiting for MCP thread to join")
                with self.mcp_lock:
                    if self.mcp_thread and self.mcp_thread.is_alive():
                        thread_ref = self.mcp_thread
                    else:
                        thread_ref = None
                
                if thread_ref:
                    thread_ref.join(timeout=1.0)  # Shorter timeout
                    debug_print("MCP thread joined")
            except Exception as e:
                debug_print(f"Error joining MCP thread: {str(e)}")
            
            with self.mcp_lock:
                self.mcp_running = False
                debug_print("MCP event loop stopped")
    
    def connect_to_mcp(self, server_script_path, callback):
        """Connect to an MCP server if specified"""
        debug_print(f"Connecting to MCP server...")
        # Store the server path for reconnection
        self.server_script_path = server_script_path
        
        # Make sure we have a running loop
        self.init_mcp_if_needed()
        
        # First check and restart loop if needed
        restarted = self._check_and_restart_mcp_loop()
        if restarted:
            debug_print("Restarted MCP loop before connection")
            time.sleep(0.3)  # Shorter wait
            
        async def connect_async():
            try:
                debug_print("Starting MCP connection...")
                is_python = server_script_path.endswith('.py')
                is_js = server_script_path.endswith('.js')
                if not (is_python or is_js):
                    raise ValueError("Server script must be a .py or .js file")

                # Clean up any existing connection
                if self.exit_stack:
                    try:
                        debug_print("Closing existing MCP connection")
                        await self.exit_stack.aclose()
                    except Exception as e:
                        debug_print(f"Error closing previous session: {str(e)}")
                    finally:
                        self.exit_stack = None
                        self.mcp_session = None
                        self.mcp_enabled = False
                
                # Initialize command based on script type
                command = "python" if is_python else "node"
                server_params = StdioServerParameters(
                    command=command,
                    args=[server_script_path],
                    env=None
                )
                
                # Set up the MCP connection
                from contextlib import AsyncExitStack
                debug_print("Creating new MCP connection")
                self.exit_stack = AsyncExitStack()
                
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport
                self.mcp_session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                
                await self.mcp_session.initialize()
                self.mcp_enabled = True
                
                # List available tools
                response = await self.mcp_session.list_tools()
                tools = response.tools
                tool_names = [tool.name for tool in tools]
                debug_print(f"MCP connected with tools: {tool_names}")
                
                # Call the callback with success result
                callback(True, f"Connected to MCP server with tools: {tool_names}")
                
            except Exception as e:
                # Call the callback with the error
                import traceback
                debug_print(f"MCP connection error: {str(e)}")
                debug_print(traceback.format_exc())
                self.mcp_enabled = False
                callback(False, str(e))
        
        # Check if we have a valid loop to use
        with self.mcp_lock:
            valid_loop = (self.mcp_loop and self.mcp_running and 
                         not self.mcp_loop.is_closed())
            
        # Schedule the connection in the MCP event loop
        if valid_loop:
            debug_print("Scheduling connection in MCP event loop")
            asyncio.run_coroutine_threadsafe(connect_async(), self.mcp_loop)
        else:
            debug_print("Cannot schedule connection - MCP event loop not running")
            callback(False, "MCP event loop not running")
            
    def process_query(self, query, message_history=None):
        """Process a query using Claude with or without MCP"""
        debug_print(f"Processing query: {query[:30]}...")
        if message_history is None:
            message_history = []
        
        # Make sure we have a running loop if MCP is enabled
        if self.mcp_enabled:
            debug_print("MCP enabled, making sure loop is running")
            self.init_mcp_if_needed()
            
            # Check and possibly restart the MCP loop
            restarted = self._check_and_restart_mcp_loop()
            if restarted:
                debug_print("Restarted MCP loop before query")
                time.sleep(0.3)  # Shorter wait
                
                # If we needed to restart, we should reconnect first
                if self.server_script_path:
                    debug_print("Reconnecting to MCP server after loop restart")
                    self.full_reset_and_reconnect()
        
        # Use a threading.Event and a shared result container for synchronization
        from threading import Event
        from queue import Queue
        result_queue = Queue(1)
        done_event = Event()
            
        # Define how to process the query based on MCP availability
        async def process_async():
            try:
                if self.mcp_enabled and self.mcp_session:
                    try:
                        debug_print("Using MCP session for query")
                        response = await asyncio.wait_for(
                            self._process_mcp_query(query, message_history),
                            timeout=25.0  # Shorter than the overall timeout to allow for graceful fallback
                        )
                        result_queue.put(response)
                    except asyncio.TimeoutError:
                        debug_print("MCP query timed out, falling back to direct query")
                        # If MCP times out, reset connection for future use and fall back to direct query
                        self.full_reset_and_reconnect()
                        response = await self._process_direct_query(query, message_history)
                        result_queue.put(response)
                    except Exception as e:
                        import traceback
                        debug_print(f"Error in MCP query, falling back to direct: {str(e)}")
                        debug_print(traceback.format_exc())
                        # If MCP fails, fall back to direct query
                        response = await self._process_direct_query(query, message_history)
                        result_queue.put(response)
                else:
                    debug_print("Using direct query (no MCP)")
                    response = await self._process_direct_query(query, message_history)
                    result_queue.put(response)
            except Exception as e:
                debug_print(f"Error in process_async: {str(e)}")
                result_queue.put(f"Error processing query: {str(e)}")
            finally:
                # Signal that we're done
                done_event.set()
        
        # Check if we have a valid MCP loop
        with self.mcp_lock:
            valid_mcp_loop = (self.mcp_enabled and self.mcp_loop and self.mcp_running and 
                             not self.mcp_loop.is_closed())
            
        # If we have MCP enabled with valid loop, use that event loop
        if valid_mcp_loop:
            # Schedule in the dedicated MCP loop
            debug_print("Scheduling query in MCP event loop")
            asyncio.run_coroutine_threadsafe(process_async(), self.mcp_loop)
        else:
            # Create a new event loop for this query
            debug_print("Creating new event loop for direct query")
            
            def run_in_new_loop():
                # Create and set the loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Run the query
                    loop.run_until_complete(process_async())
                except Exception as e:
                    debug_print(f"Error in run_in_new_loop: {str(e)}")
                    if not done_event.is_set():
                        result_queue.put(f"Error processing query: {str(e)}")
                        done_event.set()
                finally:
                    # Clean up
                    loop.close()
            
            # Run in a separate thread
            threading.Thread(target=run_in_new_loop).start()
        
        # Wait for the result with a timeout
        start_time = time.time()
        timeout = 30.0  # 30 second timeout
        
        debug_print(f"Waiting for query result with {timeout}s timeout")
        success = done_event.wait(timeout)
        if success:
            # Result is ready
            result = result_queue.get()
            debug_print(f"Got result in {time.time() - start_time:.2f}s")
            return result
        else:
            # Timeout occurred - reset the connection for future queries
            debug_print(f"Query timed out after {timeout} seconds - resetting MCP connection")
            self.full_reset_and_reconnect()
            return f"Query timed out after {timeout} seconds. Please try again."
    
    async def _process_direct_query(self, query: str, message_history) -> str:
        """Process a query using Claude directly (without MCP)"""
        debug_print("Processing direct query without MCP")
        # Create messages array from history plus current query
        messages = message_history.copy()
        
        # If the last message was from the user, replace it
        # Otherwise, add the new user message
        if not messages or messages[-1]['role'] != 'user':
            messages.append({"role": "user", "content": query})
        else:
            messages[-1] = {"role": "user", "content": query}
        
        # Call Claude API with full message history
        debug_print("Calling Claude API directly")
        try:
            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=messages
            )
            
            debug_print("Got direct response from Claude")
            return response.content[0].text
        except Exception as e:
            debug_print(f"Error in direct query: {str(e)}")
            raise
    
    async def _process_mcp_query(self, query: str, message_history) -> str:
        """Process a query using Claude with MCP tools"""
        debug_print("Starting MCP query processing")
        if not self.mcp_session:
            debug_print("Error: MCP session not available")
            return "Error: MCP session not available. Please reconnect to the MCP server."
            
        # Create messages array from history plus current query
        messages = message_history.copy()
        
        # If the last message was from the user, replace it
        # Otherwise, add the new user message
        if not messages or messages[-1]['role'] != 'user':
            messages.append({"role": "user", "content": query})
        else:
            messages[-1] = {"role": "user", "content": query}

        try:
            debug_print("Getting available tools from MCP")
            
            # Check if event loop is closed and handle accordingly
            with self.mcp_lock:
                if not self.mcp_loop or self.mcp_loop.is_closed():
                    debug_print("MCP loop is closed, cannot proceed with tools")
                    return await self._process_direct_query(query, message_history)
                
            # Get available tools from MCP with timeout
            try:
                debug_print("Listing available tools with timeout")
                response = await asyncio.wait_for(
                    self.mcp_session.list_tools(),
                    timeout=5.0  # 5 second timeout for tool listing
                )
                available_tools = [{
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in response.tools]
            except asyncio.TimeoutError:
                debug_print("Tool listing timed out, falling back to direct query")
                # Reset connection for future use
                self.full_reset_and_reconnect()
                return await self._process_direct_query(query, message_history)
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    debug_print("Event loop closed during tool listing, restarting and falling back to direct")
                    self.full_reset_and_reconnect()
                    return await self._process_direct_query(query, message_history)
                else:
                    raise

            debug_print(f"Available tools: {[t['name'] for t in available_tools]}")
            debug_print("Calling Claude API with message history and tools")
            
            # Define the system message content - DO NOT add to messages
            system_content = (
                "You have access to tools for weather information, but you should still use your general knowledge "
                "to answer questions unrelated to these tools. Only use the tools when the question is about weather, "
                "forecasts, or alerts. For all other questions like math, general knowledge, etc., answer directly "
                "you should never mention in your answers you have access to any tools. Answers like below mentioning potential tools you"
                " have access is strictly forbidden! For example: user: what is 3 + 2? assistent: 5 This is a basic addition problem" 
                "that doesn't require any special tools to solve."
                ""
            )
            
            # Initial Claude API call with system parameter (not in messages)
            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=1000,
                system=system_content,  # System instruction as a separate parameter
                messages=messages,
                tools=available_tools
            )

            debug_print("Got initial response from Claude")
            
            # Process response and handle tool calls
            tool_results = []
            final_text = []

            assistant_message_content = []
            for content in response.content:
                if content.type == 'text':
                    debug_print(f"Text content: {content.text[:50]}...")
                    final_text.append(content.text)
                    assistant_message_content.append(content)
                elif content.type == 'tool_use':
                    tool_name = content.name
                    tool_args = content.input
                    
                    debug_print(f"Tool use request: {tool_name} with args {tool_args}")

                    # Execute tool call with timeout
                    try:
                        debug_print(f"Calling tool {tool_name} with timeout")
                        result = await asyncio.wait_for(
                            self.mcp_session.call_tool(tool_name, tool_args),
                            timeout=5.0  # 5 second timeout for tool call
                        )
                        tool_results.append({"call": tool_name, "result": result})
                        final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                        
                        debug_print(f"Tool result: {result.content[:50] if result.content else 'No content'}...")

                        assistant_message_content.append(content)
                        
                        # Create updated messages for the next API call
                        updated_messages = messages.copy()
                        updated_messages.append({
                            "role": "assistant",
                            "content": assistant_message_content
                        })
                        updated_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": result.content
                                }
                            ]
                        })

                        debug_print("Getting next response from Claude")
                        # Get next response from Claude - still using system parameter
                        response = self.anthropic.messages.create(
                            model=self.model,
                            max_tokens=1000,
                            system=system_content,  # Keep using system parameter
                            messages=updated_messages,
                            tools=available_tools
                        )

                        final_text.append(response.content[0].text)
                        debug_print(f"Next response: {response.content[0].text[:50]}...")
                    except asyncio.TimeoutError:
                        debug_print(f"Tool call to {tool_name} timed out")
                        final_text.append(f"[Tool call to {tool_name} timed out]")
                        # Continue with what we have
                        break
                    except RuntimeError as e:
                        if "Event loop is closed" in str(e):
                            debug_print("Event loop closed during tool call, cannot continue with tools")
                            final_text.append(f"[Error: Could not call tool {tool_name} - MCP connection was lost]")
                            # Break out of the loop
                            break
                        else:
                            raise

            debug_print("MCP query processing complete")
            return "\n".join(final_text)
            
        except Exception as e:
            import traceback
            debug_print(f"Error in _process_mcp_query: {str(e)}")
            debug_print(traceback.format_exc())
            
            # Handle specific errors like closed event loop by restarting
            if isinstance(e, RuntimeError) and "Event loop is closed" in str(e):
                debug_print("Event loop was closed, restarting MCP loop")
                self.full_reset_and_reconnect()
                return f"The MCP connection was interrupted. Please try your question again."
                
            return f"Error processing query with MCP tools: {str(e)}"
    
    def cleanup(self):
        """Clean up resources"""
        debug_print("Cleaning up Claude resources")
        
        # Just use our reset functionality
        self._cleanup_mcp_resources()